import glob
import io
import os
from datetime import datetime
import dgl
import pandas as pd
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import optim
from torch.utils.data import DataLoader,  Dataset

from dataset import MLPReadoutlddtClass
# 定义参数
from model.LeftNet.model import *
from model.GCN import GCN
from model.utils import log

CUDA_LAUNCH_BLOCKING = 1

_version = 'Left_lddt_v3_tmp'
_init_lr = 0.0001
_weight_decay = 0.001
_criterion = 'mse'  # 使用 MSE 损失函数
_read_out = 'sum'
_seed = 42
_epochs = 100
_batch_size = 32
_lr_reduce_factor = 16
_early_stop_patience = 10
_accumulate_grad_batches = 1
_opt = 'adam'  # 使用 Adam 优化器
_dataset = 'tmp1'  # 数据集名称

# 定义特征维度
_node_input_dim = 524  # 节点特征的维度
_hidden_dim = 256
_node_out_dim = 524

# 设置日志和检查点保存目录
_wb_out_dir = './logs/'
_ckpt_out_dir = './checkpoints/'

# 确保输出目录存在
os.makedirs(_wb_out_dir, exist_ok=True)
os.makedirs(_ckpt_out_dir, exist_ok=True)

# 设定当前时间
now = datetime.now()
_CURRENT_TIME = now.strftime("%Y-%m-%d_%H-%M-%S")


class DGLDataset(Dataset):
    def __init__(self, data_folder, score_csv, file_list=None):
        """
        初始化 DGLDataset
        :param data_folder: 数据集所在文件夹路径
        :param score_csv: 包含评分数据的 CSV 文件路径
        :param file_list: 文件名列表（可选），如果指定则只加载该列表中的文件
        """
        self.data_folder = data_folder
        self.score_data = self.load_score_data(score_csv)

        if file_list:
            # 如果提供了文件名列表，只加载该列表中的文件
            self.dgl_files = [os.path.join(data_folder, f"{file_name}") for file_name in file_list]
        else:
            # 否则加载文件夹中的所有 .dgl 文件
            self.dgl_files = sorted(glob.glob(os.path.join(data_folder, '*.dgl')))

    def __len__(self):
        return len(self.dgl_files)

    def __getitem__(self, idx):
        dgl_file = self.dgl_files[idx]
        dgl_graphs, _ = dgl.load_graphs(dgl_file)

        if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
            dgl_graph = dgl_graphs[0]
        else:
            raise ValueError(f"No DGL graph loaded from {dgl_file}")

        # 获取全局 decoy 索引的评分
        score = self.get_score(idx + 1)  # 假设 idx 是全局索引

        return dgl_graph, score

    def load_score_data(self, score_csv):
        """
        加载评分数据，假设每个文件在 CSV 中有 'Decoy Index' 和 'lDDT' 两列
        """
        with open(score_csv, 'r') as f:
            content = f.read()

        parts = content.split("=== Contents of ")
        score_data = {}
        global_index = 1

        for part in parts[1:]:  # 跳过第一个部分，因为它不包含有用数据
            lines = part.strip().split('\n')
            file_name = lines[0].split('.csv')[0]
            csv_content = '\n'.join(
                [line for line in lines[1:] if 'NULL' not in line and line.strip()])  # 过滤掉包含 NULL 的行和空行

            # 读取 CSV 内容并跳过无用行
            csv_io = io.StringIO(csv_content)
            df = pd.read_csv(csv_io)

            # 检查列名是否存在
            if 'Decoy Index' not in df.columns or 'lDDT' not in df.columns:
                raise KeyError(f"Required columns are missing in {file_name}")

            # 将每个文件的 Decoy Index 转换为全局索引
            for local_index in df['Decoy Index']:
                score_data[global_index] = df.loc[df['Decoy Index'] == local_index, 'lDDT'].values[0]
                global_index += 1

        return score_data

    def get_score(self, decoy_index):
        """
        根据全局索引获取对应的 lDDT 评分
        """
        if decoy_index in self.score_data:
            return self.score_data[decoy_index]
        else:
            raise ValueError(f"No score found for Decoy Index {decoy_index}")


class QAModel(pl.LightningModule):
    def __init__(self):
        super(QAModel, self).__init__()

        self.node_input_dim = _node_input_dim
        self.hidden_dim = _hidden_dim
        self.node_out_dim = _node_out_dim
        self.readout = _read_out
        self.init_lr = _init_lr
        self.weight_decay = _weight_decay
        self.opt = _opt
        if _criterion == 'mse':
            print('USE MSE')
            self.criterion = torchmetrics.MeanSquaredError()
        elif _criterion == 'mae':
            print('USE MAE')
            self.criterion = torchmetrics.MeanAbsoluteError()
        else:
            print('DEFAULT IS MSE')
            self.criterion = torchmetrics.MeanSquaredError()

        # GCN 参数
        self.gcn = GCN(self.node_input_dim, self.hidden_dim, self.node_out_dim)

        # LeftNet参数
        self.leftnet = LEFTNet(
            pos_require_grad=False, cutoff=8.0, num_layers=4,
            hidden_channels=128, num_radial=32, y_mean=0, y_std=1
        )
        self.last_layer = nn.Linear(self.node_out_dim, 1)
        self.mlp_readout = MLPReadoutlddtClass(input_dim=self.node_input_dim, output_dim=1, dp_rate=0.5)

        # 初始化用于保存训练和验证损失的列表
        self.train_losses = []
        self.val_losses = []

    # 输入特征 h、坐标 x
    def forward(self, g, h, x, batch):
        """
        :param batch:
        :param g: DGL graph object
        :param h: Initial node features (scalar features)
        :param x: Node coordinates (vector features)
        """
        h = torch.nan_to_num(h, nan=0.0)
        x = torch.nan_to_num(x, nan=0.0)

        # Step 1: 通过 GCN 层来更新节点特征
        # h = self.gcn(g, h)

        # Step 2: 准备 LeftNet 输入
        h = h.to(self.device)
        pos = x  # 假设 x 是节点的 CA 坐标
        pos = pos.to(self.device)  # 确保 pos 在正确的设备上
        batch = batch.to(self.device)  # 确保 batch 在正确的设备上

        batch_data = {
            'x': h,  # 更新后的节点特征
            'coords_ca': pos,  # Cα坐标
            'batch': batch,  # 批次信息
        }

        # Step 3: 通过 LeftNet 获取预测
        leftnet_output = self.leftnet(batch_data)

        # 将更新后的节点特征存储到图中，键改为 'feat'
        g.ndata['feat'] = leftnet_output

        # Step 6: 图级特征汇聚操作（Readout），键改为 'feat'
        if self.readout == "sum":
            zg = dgl.sum_nodes(g, 'feat')  # zg shape: [batch_size, feature_dim]
        elif self.readout == "max":
            zg = dgl.max_nodes(g, 'feat')
        elif self.readout == "mean":
            zg = dgl.mean_nodes(g, 'feat')
        else:
            zg = dgl.mean_nodes(g, 'feat')  # 默认为mean

        # Step 7: 通过MLP层（多层感知机）进行最终预测
        pred_lddt = self.mlp_readout(zg)  # 期望 pred_lddt 形状为 [batch_size, 1]

        return pred_lddt

    def configure_optimizers(self):
        if self.opt == 'adam':
            print('USING ADAM')
            optimizer = optim.Adam(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay, amsgrad=True)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay, momentum=0.9,
                                  nesterov=True)

        if self.current_epoch < 20:
            return {
                'optimizer': optimizer,
                "lr_scheduler": {
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=_lr_reduce_factor, gamma=0.5),
                    'monitor': 'val_loss'
                }
            }
        else:
            print('START to USE SGD OPTIMIZER!')
            optimizer = optim.SGD(self.parameters(), lr=self.init_lr * 0.1, weight_decay=self.weight_decay,
                                  momentum=0.9, nesterov=True)
            return {
                'optimizer': optimizer,
                "lr_scheduler": {
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=_lr_reduce_factor, gamma=0.5),
                    'monitor': 'val_loss'
                }
            }

    def training_step(self, train_batch, batch_idx):
        # 从 DGL 图中提取节点特征、坐标和目标
        dgl_graph, batch_targets = train_batch

        batch_sizes = dgl_graph.batch_num_nodes()
        node_batch = torch.cat([torch.full((n,), i) for i, n in enumerate(batch_sizes)])

        # 提取节点特征
        batch_n = dgl_graph.ndata['feat'].squeeze()
        coords = dgl_graph.ndata['coords'].squeeze()

        # 获取 batch 信息
        batch_size = dgl_graph.batch_size  # 获取批次大小

        # 模型前向传播
        pred_lddt = self.forward(dgl_graph, batch_n, coords, node_batch)

        # lddt loss
        lddt_loss = self.criterion(pred_lddt, batch_targets)

        train_loss = lddt_loss

        # 记录训练损失
        self.train_losses.append(train_loss.item())

        # log
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # 使用wandb记录训练损失
        import wandb
        wandb.log({'train_loss': train_loss.item(), 'epoch': self.current_epoch})

        # 输出训练进度和损失值，每隔10个batch输出一次
        if batch_idx % 10 == 0:
            log(f"{batch_idx}/{self.current_epoch} train_epoch ||| Loss: {round(float(train_loss), 6)}")

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        dgl_graph, batch_targets = val_batch
        # print(f"val_batch: {val_batch}")

        batch_sizes = dgl_graph.batch_num_nodes()
        node_batch = torch.cat([torch.full((n,), i) for i, n in enumerate(batch_sizes)])

        # 提取节点特征
        batch_n = dgl_graph.ndata['feat'].squeeze()
        coords = dgl_graph.ndata['coords'].squeeze()

        # 获取 batch 信息
        batch_size = dgl_graph.batch_size  # 获取批次大小

        # 模型前向传播
        pred_lddt = self.forward(dgl_graph, batch_n, coords, node_batch)

        # lddt loss
        lddt_loss = self.criterion(pred_lddt, batch_targets)

        # 打印损失值
        print(f"lDDT Loss: {lddt_loss.item()}")

        val_loss = lddt_loss

        # 记录验证损失
        self.val_losses.append(val_loss.item())

        # log
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # 使用wandb记录验证损失
        import wandb
        wandb.log({'val_loss': val_loss.item(), 'epoch': self.current_epoch})

        return val_loss

    def on_train_end(self):
        print("Training is finished. Saving losses...")
        # 在训练结束时保存损失列表
        save_path = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\losses\losses_left_lddt_tmp_3.json"

        losses_dict = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }

        with open(save_path, 'w') as f:
            f.write("{\n")
            f.write(f'  "train_losses": {self.train_losses},\n')
            f.write(f'  "val_losses": {self.val_losses}\n')
            f.write("}\n")
        print(f"训练和验证损失保存到 {save_path}")


# 自定义 collate_fn
def collate_fn(batch):
    graphs, scores = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)

    batched_scores = torch.tensor(scores).unsqueeze(1)

    return batched_graph, batched_scores


def load_dataset_from_txt(trainset_path, valset_path):
    # 读取训练集文件名
    with open(trainset_path, 'r') as train_f:
        train_files = [line.strip() for line in train_f.readlines()]

    # 读取验证集文件名
    with open(valset_path, 'r') as val_f:
        val_files = [line.strip() for line in val_f.readlines()]

    return train_files, val_files


# 创建数据集
data_folder = r'D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\DGL\Final_DGL'  # 替换为实际的数据集文件夹路径
score_csv = r'D:\pycharm\pycharmProjects\ideamodel3qa\data\data\train\Final_tmp_lddt.csv'
dataset = DGLDataset(data_folder, score_csv)

# 划分训练集和验证集
train_files, val_files = load_dataset_from_txt('../data/data/dataset_txt/trainset_tmp.txt',
                                               '../data/data/dataset_txt/valset_tmp.txt')

# 加载训练集和验证集
train_dataset = DGLDataset(data_folder, score_csv, file_list=train_files)
val_dataset = DGLDataset(data_folder, score_csv, file_list=val_files)

# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, persistent_workers=True,
                          shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, persistent_workers=True,
                        shuffle=False, pin_memory=True, collate_fn=collate_fn)

model = QAModel().to('cuda')  # 显式地将模型移动到 GPU


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)

pl.seed_everything(_seed)

# logger
wandb_logger = WandbLogger(project="LeftQA",
                           name=_version,
                           id=_CURRENT_TIME,
                           offline=False,
                           log_model=False,
                           save_dir=_wb_out_dir)

# training
# define callbacks
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    patience=_early_stop_patience,
                                    verbose=True,
                                    mode="min")

lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(_ckpt_out_dir, _version),
    filename='{epoch}-{val_loss:.5f}',  # 验证损失应该自动由 val_loss 提供
    monitor='val_loss',  # 监控验证损失
    save_top_k=3,  # 只保存最好的三个模型
    mode='min',  # 最小化损失
    save_weights_only=True,  # 仅保存权重
    verbose=True  # 显示详细保存信息
)

saw = StochasticWeightAveraging(swa_epoch_start=0.7,
                                swa_lrs=_init_lr * 0.1,
                                annealing_epochs=10,
                                annealing_strategy='cos')

# define a trainer
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,  # 使用单个GPU
    num_nodes=1,
    max_epochs=_epochs,
    logger=wandb_logger,  # 设置为 None，禁用任何日志记录
    callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, saw],
    sync_batchnorm=True,
    accumulate_grad_batches=_accumulate_grad_batches
)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)

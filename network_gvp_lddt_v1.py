import io
import json
import os
from datetime import datetime

import dgl
import numpy as np
import pandas as pd
import torch
import torch_scatter
from Bio.PDB import PDBParser
from torch import nn
from torch import optim
from torch.cuda import device
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import glob
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from EGNN import EGNN
from GVP_GNN import GVPConvLayer, _normalize, LayerNorm, GVP
from dataset import MLPReadoutlddtClass

# 定义参数

_version = 'GVP_lddt_v1'
_init_lr = 0.0001
_weight_decay = 5e-4
_criterion = 'mse'  # 使用 MSE 损失函数
_read_out = 'sum'
_seed = 42
_epochs = 100
_batch_size = 32
_lr_reduce_factor = 16
_early_stop_patience = 15
_accumulate_grad_batches = 5
_opt = 'adam'  # 使用 Adam 优化器
_dataset = 'MUL_tmp'  # 数据集名称

# 定义特征维度
_node_input_dim = 300  # 节点特征的维度
_hidden_dim = 256
_node_out_dim = 300
_edge_input_dim = 1  # 边特征的维度
_node_scalar_feature_dim = 300  # 节点标量特征维度
_node_vector_feature_dim = 3  # 节点矢量特征维度
_edge_scalar_feature_dim = 1  # 边的标量特征维度
_edge_vector_feature_dim = 1  # 边的矢量特征维度

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
    def __init__(self, dgl_folder, pdb_folder, score_csv):
        self.dgl_folder = dgl_folder
        self.pdb_folder = pdb_folder
        self.score_data = self.load_score_data(score_csv)
        self.dgl_files = sorted(glob.glob(os.path.join(dgl_folder, '*.dgl')))
        self.pdb_files = sorted(glob.glob(os.path.join(pdb_folder, '*.pdb')))

        if len(self.dgl_files) != len(self.pdb_files):
            raise ValueError("The number of DGL files and PDB files must be the same.")

        # Create a mapping of DGL files to PDB files based on their names
        self.file_mapping = {
            os.path.splitext(os.path.basename(dgl_file))[0]: os.path.splitext(os.path.basename(pdb_file))[0]
            for dgl_file, pdb_file in zip(self.dgl_files, self.pdb_files)
        }

    def __len__(self):
        return len(self.dgl_files)

    def __getitem__(self, idx):
        dgl_file = self.dgl_files[idx]
        dgl_file_name = os.path.splitext(os.path.basename(dgl_file))[0]

        # 根据DGL文件名找到对应的PDB文件
        if dgl_file_name in self.file_mapping:
            pdb_file_name = self.file_mapping[dgl_file_name]
            pdb_file = os.path.join(self.pdb_folder, f"{pdb_file_name}.pdb")
        else:
            raise ValueError(f"No corresponding PDB file found for {dgl_file}")

        # 加载DGL图
        dgl_graphs, _ = dgl.load_graphs(dgl_file)

        if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
            dgl_graph = dgl_graphs[0]
        else:
            raise ValueError(f"No DGL graph loaded from {dgl_file}")

        # 加载 PDB 文件并提取坐标
        coords = self.load_pdb_coords(pdb_file)

        # 获取目标分数
        score = self.get_score(idx + 1)  # 假设 idx 是全局索引

        # 返回 DGL 图、坐标、分数
        return dgl_graph, coords, score

    def load_score_data(self, score_csv):
        with open(score_csv, 'r') as f:
            content = f.read()

        parts = content.split("=== Contents of ")
        score_data = {}
        global_index = 1

        for part in parts[1:]:  # Skip the first part as it doesn't contain useful data
            lines = part.strip().split('\n')
            file_name = lines[0].split('.csv')[0]
            csv_content = '\n'.join(
                [line for line in lines[1:] if 'NULL' not in line and line.strip()])  # 过滤掉包含 NULL 的行和空行

            # 读取csv内容并跳过无用的行
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

    def load_pdb_coords(self, pdb_file):
        # 创建 PDB 解析器
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('PDB_structure', pdb_file)

        # 存储坐标
        coordinates = []

        # 遍历每个模型、链和残基
        for model in structure:
            for chain in model:
                for residue in chain:
                    # 提取 CA 原子的坐标
                    if residue.has_id('CA'):
                        coordinates.append(residue['CA'].get_coord())

        # 转换为 NumPy 数组
        coords = np.array(coordinates)

        return coords

    def get_score(self, decoy_index):
        if decoy_index in self.score_data:
            return self.score_data[decoy_index]
        else:
            raise ValueError(f"No score found for Decoy Index {decoy_index}")


class QAModel(pl.LightningModule):
    def __init__(self, node_input_dim, hidden_dim, node_out_dim, edge_input_dim, node_scalar_feature_dim,
                 node_vector_feature_dim, edge_scalar_feature_dim, edge_vector_feature_dim, read_out,
                 init_lr, weight_decay, opt):
        super(QAModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.node_out_dim = node_out_dim
        self.edge_input_dim = edge_input_dim
        self.node_scalar_feature_dim = node_scalar_feature_dim
        self.node_vector_feature_dim = node_vector_feature_dim
        self.edge_scalar_feature_dim = edge_scalar_feature_dim
        self.edge_vector_feature_dim = edge_vector_feature_dim
        self.readout = read_out
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.opt = opt

        # 定义线性层用于降维
        self.dim_reduction_layer = nn.Linear(self.node_input_dim, 128)

        # EGNN 参数
        self.egnn = EGNN(
            in_node_nf=300,  # 输入节点特征维度
            hidden_nf=self.hidden_dim,  # 隐藏特征维度
            out_node_nf=self.node_out_dim,  # 输出节点特征维度
            in_edge_nf=self.edge_input_dim,  # 输入边特征维度
            device='cuda' if torch.cuda.is_available() else 'cpu',
            act_fn=nn.SiLU(),  # 激活函数
            n_layers=4,  # EGNN层数
            residual=True,  # 是否用残差连接
            attention=True,  # 是否使用注意力机制
            normalize=False,  # 是否归一化
            tanh=False  # 输出是否使用tanh
        )

        # GVP embedding for edges and nodes
        self.W_e = nn.Sequential(
            LayerNorm((self.edge_scalar_feature_dim, 1)),
            GVP((self.edge_scalar_feature_dim, 1), (self.edge_scalar_feature_dim, 1),
                activations=(None, None), vector_gate=True)
        )

        self.W_v = nn.Sequential(
            LayerNorm((self.node_scalar_feature_dim, 0)),
            GVP((self.node_scalar_feature_dim, 0), (self.node_scalar_feature_dim, self.node_vector_feature_dim),
                activations=(None, None), vector_gate=True)
        )

        # GVPConvLayer 参数
        self.gvp = nn.ModuleList(GVPConvLayer(
            node_dims=(self.node_scalar_feature_dim, self.node_vector_feature_dim),  # 节点特征维度
            edge_dims=(self.edge_scalar_feature_dim, self.edge_vector_feature_dim),  # 边特征维度
            activations=(F.relu, torch.sigmoid),  # 激活函数（标量，矢量）
            vector_gate=True  # 是否使用矢量门控
        ) for _ in range(5))

        self.W_out = nn.Sequential(LayerNorm((self.node_scalar_feature_dim, 3)),
                                   GVP((self.node_scalar_feature_dim, 3),
                                       (self.node_scalar_feature_dim, 0),
                                       activations=(F.relu, torch.sigmoid), vector_gate=True))

        self.mlp_readout = MLPReadoutlddtClass(input_dim=self.node_input_dim, output_dim=1, dp_rate=0.5)

        if _criterion == 'mse':
            print('USE MSE')
            self.criterion = torchmetrics.MeanSquaredError()
        elif _criterion == 'mae':
            print('USE MAE')
            self.criterion = torchmetrics.MeanAbsoluteError()
        else:
            print('DEFAULT IS MSE')
            self.criterion = torchmetrics.MeanSquaredError()

        # 初始化用于保存训练和验证损失的列表
        self.train_losses = []
        self.val_losses = []

    # 输入特征 h、坐标 x、边索引 edges 和边特征 edge_attr
    def forward(self, g, h, x, edge_index, edge_attr, scatter_mean=True, dense=True):
        """
        Forward pass through EGNN and GVPConvLayer
        :param g: DGL graph object
        :param h: Initial node features (scalar features)
        :param x: Node coordinates (vector features)
        :param edge_index: Edge indices (from DGL graph)
        :param edge_attr: Edge attributes (features)
        """
        h = torch.nan_to_num(h, nan=0.0)
        x = torch.nan_to_num(x, nan=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

        # h = self.dim_reduction_layer(h)  # 将 h 的维度降到 128
        # Step 1: 通过EGNN层来更新节点特征和坐标
        h, x = self.egnn(h, x, edge_index, edge_attr)

        # 打印输入 edge_attr 的形状，确保其维度是正确的
        # print(f"Edge attr shape before EGNN: {edge_attr.shape}")

        # Step 2: 通过 W_e 和 W_v 预处理边和节点特征
        # 计算边的矢量特征：终点节点坐标 - 起点节点坐标
        edge_vectors = x[edge_index[0]] - x[edge_index[1]]  # 形状为 [n_edges, 3]
        edge_vectors_normalized = _normalize(edge_vectors).unsqueeze(-2)  # 变为 [n_edges, 1, 3]
        # print(f"Edge vector shape: {edge_vectors_normalized.shape}")
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        edge_vectors_normalized = torch.nan_to_num(edge_vectors_normalized, nan=0.0)

        # 将标量和矢量边特征组合起来，形成 (标量特征, 矢量特征) 的元组
        edge_inputs = (edge_attr, edge_vectors_normalized)

        # Step 3: 准备GVP层的边输入
        edge_inputs = self.W_e(edge_inputs)

        # Step 4: 准备GVP层的节点输入
        node_inputs = self.W_v(h)

        # Step 5: 通过GVP层更新节点特征
        for layer in self.gvp:
            node_inputs = layer(node_inputs, edge_index, edge_inputs)

        # 提取更新后的标量节点特征 (h)
        updated_scalar_features = self.W_out(node_inputs)

        # 将更新后的节点特征存储到图中，键改为 'feat'
        g.ndata['feat'] = updated_scalar_features

        # Step 6: 图级特征汇聚操作（Readout），键改为 'feat'
        if self.readout == "sum":
            zg = dgl.sum_nodes(g, 'feat')
        elif self.readout == "max":
            zg = dgl.max_nodes(g, 'feat')
        elif self.readout == "mean":
            zg = dgl.mean_nodes(g, 'feat')
        else:
            zg = dgl.mean_nodes(g, 'feat')  # 默认为mean

        # Step 6: 通过MLP层（多层感知机）进行最终预测
        pred_lddt = self.mlp_readout(zg)

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
        dgl_graph, coords, batch_targets = train_batch

        # 提取节点特征
        batch_n = dgl_graph.ndata['feat'].squeeze()
        coords = coords.squeeze()

        batch_n_min = batch_n.min()
        batch_n_max = batch_n.max()
        if batch_n_min == batch_n_max:
            print("batch_min and batch_max are equal. This will lead to division by zero.")
        batch_n = (batch_n - batch_n_min) / (batch_n_max - batch_n_min + 1e-8) * 2 - 1  # 归一化到 [-1, 1]

        # 提取源节点索引 (src_index) 和目标节点索引 (dst_index)
        src_index, dst_index = dgl_graph.edges()

        # 分别对 src_index 和 dst_index 进行 squeeze 操作
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()

        # 将 src_index 和 dst_index 堆叠成形状为 [2, num_edges] 的张量
        edge_index = torch.stack([src_index, dst_index], dim=0)

        # 提取边特征
        edge_attr = dgl_graph.edata['weight'].squeeze()
        edge_attr = edge_attr.unsqueeze(dim=1)  # 根据需要调整维度

        # 模型前向传播
        pred_lddt = self.forward(dgl_graph, batch_n, coords, edge_index, edge_attr)
        # 获取当前批次大小
        batch_size = dgl_graph.batch_size

        # lddt loss
        lddt_loss = F.mse_loss(pred_lddt.squeeze(), batch_targets.float())

        train_loss = lddt_loss

        # 打印梯度
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}")  # 打印每个参数梯度的范数
        #     else:
        #         print(f"Gradient for {name}: None")  # 如果梯度为空
        # 记录训练损失
        self.train_losses.append(train_loss.item())

        # log
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return train_loss

    def validation_step(self, val_batch, batch_idx):
        dgl_graph, coords, batch_targets = val_batch

        # 提取节点特征
        batch_n = dgl_graph.ndata['feat'].squeeze()
        coords = coords.squeeze()

        batch_n_min = batch_n.min()
        batch_n_max = batch_n.max()
        if batch_n_min == batch_n_max:
            print("batch_min and batch_max are equal. This will lead to division by zero.")
        batch_n = (batch_n - batch_n_min) / (batch_n_max - batch_n_min + 1e-8) * 2 - 1  # 归一化到 [-1, 1]

        # 提取源节点索引 (src_index) 和目标节点索引 (dst_index)
        src_index, dst_index = dgl_graph.edges()

        # 分别对 src_index 和 dst_index 进行 squeeze 操作
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()

        # 将 src_index 和 dst_index 堆叠成形状为 [2, num_edges] 的张量
        edge_index = torch.stack([src_index, dst_index], dim=0)
        # 提取边特征
        edge_attr = dgl_graph.edata['weight'].squeeze()
        edge_attr = edge_attr.unsqueeze(dim=1)  # 根据需要调整维度

        # 模型前向传播
        pred_lddt = self.forward(dgl_graph, batch_n, coords, edge_index, edge_attr)

        # 获取当前批次大小
        batch_size = dgl_graph.batch_size

        # lddt loss
        lddt_loss = F.mse_loss(pred_lddt.squeeze(), batch_targets.float())

        # 打印损失值
        print(f"lDDT Loss: {lddt_loss.item()}")

        val_loss = lddt_loss

        # 记录验证损失
        self.val_losses.append(val_loss.item())

        # log
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return val_loss

    def on_train_end(self):
        # 在训练结束时保存损失列表
        save_path = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\losses\losses_gvp_lddt_v1.json"
        with open(save_path, 'w') as f:
            json.dump({"train_losses": self.train_losses, "val_losses": self.val_losses}, f)
        print(f"训练和验证损失保存到 {save_path}")


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values!")


# 自定义 collate_fn
def collate_fn(batch):
    graphs, coords, scores = map(list, zip(*batch))

    # 将图批次化
    batched_graph = dgl.batch(graphs)

    # 将所有的坐标拼接成一个 Tensor
    batched_coords = torch.tensor(np.concatenate(coords, axis=0), dtype=torch.float32)

    # 将目标分数批次化
    batched_scores = torch.tensor(scores, dtype=torch.float)

    return batched_graph, batched_coords, batched_scores


# 创建数据集
data_folder = r'D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\MUL_DGL'  # 替换为实际的数据集文件夹路径
pdb_folder = r'D:\pycharm\pycharmProjects\ideamodel3qa\data\data\train\MUL_tmp'
score_csv = r'D:\pycharm\pycharmProjects\ideamodel3qa\data\data\train\MUL_tmp_lddt.csv'
dataset = DGLDataset(data_folder, pdb_folder, score_csv)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))  # 假设训练集占总数据集的80%
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4,
                          collate_fn=collate_fn, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=collate_fn, persistent_workers=True)

model = QAModel(
    node_input_dim=_node_input_dim,
    hidden_dim=_hidden_dim,
    node_out_dim=_node_out_dim,
    edge_input_dim=_edge_input_dim,
    node_scalar_feature_dim=_node_scalar_feature_dim,
    node_vector_feature_dim=_node_vector_feature_dim,
    edge_scalar_feature_dim=_edge_scalar_feature_dim,
    edge_vector_feature_dim=_edge_vector_feature_dim,
    read_out=_read_out,
    init_lr=_init_lr,
    weight_decay=_weight_decay,
    opt=_opt
).to('cuda')  # 显式地将模型移动到 GPU

pl.seed_everything(_seed)

# logger
# 删除日志记录部分

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
    logger=None,  # 设置为 None，禁用任何日志记录
    callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, saw],
    sync_batchnorm=True,
    accumulate_grad_batches=_accumulate_grad_batches,
    gradient_clip_val=3.0,  # 设置裁剪阈值
    gradient_clip_algorithm='norm'  # 或 'value'
)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)

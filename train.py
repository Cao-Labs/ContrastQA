import json
import os
import dgl
import numpy as np
import torch
import wandb
from datetime import datetime
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import optim, nn
from torch.utils.data import DataLoader
from dataset import MLPReadoutlddtClass, GCLDGLDataset
from GVP_GNN import GVPConvLayer, _normalize, LayerNorm, GVP
from model.utils import log

CUDA_LAUNCH_BLOCKING = 1

# load config.json
with open('./config/pre_train_knn10_seed42.json', 'r') as f:
    config = json.load(f)

# parameter
_version = config['version']
_init_lr = config["init_lr"]
_weight_decay = config["weight_decay"]
_criterion = config["criterion"]
_read_out = config["read_out"]
_mlp_dp_rate = config["mlp_dp_rate"]
_mse_weight = config["mse_weight"]
_ce_weight = config["ce_weight"]
_cl_weight = config["cl_weight"]
_seed = config["seed"]
_epochs = config["epochs"]
_batch_size = config["batch_size"]
_lr_reduce_factor = config["lr_reduce_factor"]
_early_stop_patience = config["early_stop_patience"]
_accumulate_grad_batches = config["accumulate_grad_batches"]
_opt = config["optimizer"]
_dataset = config["dataset"]
_node_input_dim = config["node_input_dim"]
_node_gvp_input_dim = config["node_gvp_input_dim"]
_node_out_dim = config["node_out_dim"]
_node_scalar_feature_dim = config["node_scalar_feature_dim"]
_node_vector_feature_dim = config["node_vector_feature_dim"]
_edge_scalar_feature_dim = config["edge_scalar_feature_dim"]
_edge_vector_feature_dim = config["edge_vector_feature_dim"]
_gvp_num_layer = config["gvp_num_layer"]

print(f"Training with version: {_version}, learning rate: {_init_lr}")

# logging file & weight save path
_wb_out_dir = './logs/trainwb'
_ckpt_out_dir = './checkpoints/'

os.makedirs(_wb_out_dir, exist_ok=True)
os.makedirs(_ckpt_out_dir, exist_ok=True)

# time
now = datetime.now()
_CURRENT_TIME = now.strftime("%Y-%m-%d_%H-%M-%S")


class QAModel(pl.LightningModule):
    def __init__(self):
        super(QAModel, self).__init__()

        self.node_input_dim = _node_input_dim
        self.node_gvp_input_dim = _node_gvp_input_dim
        self.node_out_dim = _node_out_dim
        self.node_scalar_feature_dim = _node_scalar_feature_dim
        self.node_vector_feature_dim = _node_vector_feature_dim
        self.edge_scalar_feature_dim = _edge_scalar_feature_dim
        self.edge_vector_feature_dim = _edge_vector_feature_dim
        self.node_in_dim = (self.node_gvp_input_dim, 0)
        self.node_hidden_dim = (self.node_scalar_feature_dim, self.node_vector_feature_dim)
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
        self.criterion_acc = torchmetrics.Accuracy("multiclass", num_classes=4)

        pl.seed_everything(_seed)

        # model components
        self.mse_weight = _mse_weight
        self.ce_weight = _ce_weight
        self.cl_weight = _cl_weight

        # model_feature
        self.feat_model = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_gvp_input_dim),
            nn.ReLU(),
            nn.LayerNorm(self.node_gvp_input_dim)
        )

        # GVP embedding for edges and nodes
        self.W_e = nn.Sequential(
            LayerNorm((self.edge_scalar_feature_dim, 1)),
            GVP((self.edge_scalar_feature_dim, 1), (self.edge_scalar_feature_dim, 1),
                activations=(None, None), vector_gate=True)
        )

        self.W_v = nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim,
                self.node_hidden_dim,
                activations=(None, None), vector_gate=True)
        )

        # GVPConvLayer parameter
        self.gvp = nn.ModuleList(GVPConvLayer(
            node_dims=self.node_hidden_dim,  # node feature dimension
            edge_dims=(self.edge_scalar_feature_dim, self.edge_vector_feature_dim),  # edge feature dimension
            activations=(F.relu, None),  # act(scalars, vectors)
            vector_gate=True  # vector gate 
        ) for _ in range(_gvp_num_layer))

        ns, _ = (
            self.node_hidden_dim  # Extract the scalar feature count from node hidden dimensions
        )
        self.W_out = nn.Sequential(LayerNorm(self.node_hidden_dim),
                                   GVP(self.node_hidden_dim,
                                       (ns, 0),
                                       activations=(F.relu, None), vector_gate=True))

        self.dense = nn.Sequential(
            nn.Linear(ns, ns),
            nn.ReLU(inplace=True),  # Define a linear layer and ReLU activation
            nn.Dropout(p=0.1),  # Add a Dropout layer
        )

        self.mlp_readout = MLPReadoutlddtClass(input_dim=self.node_out_dim, output_dim=1, dp_rate=_mlp_dp_rate)

        # save losses
        self.train_losses = []  
        self.val_losses = []  

        self.train_mse = []  # training MSE
        self.train_ce = []  # training CE
        self.train_cl_loss = []  # training contrast loss
        self.train_acc = []  # training acc
        self.val_mse = []  # val MSE
        self.val_ce = []  # val CE
        self.val_cl_loss = []  # val contrast loss
        self.val_acc = []  # val acc

    def nt_xent_loss(self, anchor, positive, negatives, anchor_score, neg_scores, anchor_target, neg_targets,
                     temperature=0.07):
        """
        caculate NT-Xent contrast loss based on label weight.

        :param anchor: anchor embedding
        :param positive: positive embedding
        :param negatives: negative embedding (list)
        :param neg_scores:  DockQ score corresponding to negative samples (list)
        :param anchor_score: DockQ score corresponding to anchor sample
        :param neg_targets: target name of negative samples(list)
        :param anchor_target: target name of anchor sample
        :param temperature: temperature
        :return: contrast loss
        """

        # similarity between anchor and positive samples
        pos_similarity = torch.exp(torch.cosine_similarity(anchor, positive) / temperature)

        neg_similarities = []
        neg_weights = []

        for negative, neg_score, neg_target in zip(negatives, neg_scores, neg_targets):
            # similarity between anchor and negative samples
            neg_similarity = torch.exp(torch.cosine_similarity(anchor, negative) / temperature)
            neg_similarities.append(neg_similarity)

            # weight calculating：score difference under the same target ，different targets setting to 1
            weight = torch.abs(anchor_score - neg_score) if neg_target == anchor_target else torch.ones_like(
                anchor_score)
            weight = weight.squeeze()
            neg_weights.append(weight)

        # stack all negative similarities & weights
        neg_similarities = torch.stack(neg_similarities)  # (num_negatives,)
        neg_weights = torch.stack(neg_weights)  # (num_negatives,)

        weighted_neg_sum = (neg_similarities * neg_weights).sum(dim=0)

        # loss calculate
        loss = -torch.log(pos_similarity / (pos_similarity + weighted_neg_sum)).mean()

        return loss

    def forward(self, anchor_graph, positive_graph, negative_graphs, anchor_score, neg_scores, anchor_target,
                neg_targets, scatter_mean=True, dense=True):

        # anchor embedding
        anchor_h = anchor_graph.ndata['feat'].to(self.device).float()
        anchor_x = anchor_graph.ndata['coords'].to(self.device).float()
        anchor_o = anchor_graph.ndata['ori'].to(self.device).float()
        anchor_e = anchor_graph.edata['feat'].to(self.device).float()
        anchor_h = self.feat_model(anchor_h)

        src_index, dst_index = anchor_graph.edges()
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()
        anchor_edge_index = torch.stack([src_index, dst_index], dim=0)
        anchor_edge_vectors = anchor_x[anchor_edge_index[0]] - anchor_x[anchor_edge_index[1]]  # 形状为 [n_edges, 3]
        anchor_edge_vectors_normalized = _normalize(anchor_edge_vectors).unsqueeze(-2)  # 变为 [n_edges, 1, 3]
        anchor_edge_inputs = (anchor_e, anchor_edge_vectors_normalized)
        anchor_edge_inputs = self.W_e(anchor_edge_inputs)  # Process edge features using the GVP layer
        anchor_node_inputs = self.W_v(anchor_h)  # Process node features using the GVP layer
        # residual connection
        tensor1, tensor2 = anchor_node_inputs
        tensor2 = tensor2 + anchor_o  # add ori feature to tensor2 
        anchor_node_inputs = (tensor1, tensor2)

        for layer in self.gvp:
            anchor_node_inputs = layer(anchor_node_inputs, anchor_edge_index, anchor_edge_inputs)
        anchor = self.W_out(anchor_node_inputs)
        anchor_z = self.dense(anchor)

        # positive embedding
        positive_h = positive_graph.ndata['feat'].to(self.device).float()
        positive_x = positive_graph.ndata['coords'].to(self.device).float()
        positive_o = positive_graph.ndata['ori'].to(self.device).float()
        positive_e = positive_graph.edata['feat'].to(self.device).float()
        positive_h = self.feat_model(positive_h)

        src_index, dst_index = positive_graph.edges()
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()
        positive_edge_index = torch.stack([src_index, dst_index], dim=0)
        positive_edge_vectors = positive_x[positive_edge_index[0]] - positive_x[
            positive_edge_index[1]]  # 形状为 [n_edges, 3]
        positive_edge_vectors_normalized = _normalize(positive_edge_vectors).unsqueeze(-2)  # 变为 [n_edges, 1, 3]
        positive_edge_inputs = (positive_e, positive_edge_vectors_normalized)
        positive_edge_inputs = self.W_e(positive_edge_inputs)
        positive_node_inputs = self.W_v(positive_h)
        tensor1, tensor2 = positive_node_inputs
        tensor2 = tensor2 + positive_o  
        positive_node_inputs = (tensor1, tensor2)
        for layer in self.gvp:
            positive_node_inputs = layer(positive_node_inputs, positive_edge_index, positive_edge_inputs)
        positive = self.W_out(positive_node_inputs)
        positive_z = self.dense(positive)

        # negatives embedding
        negative_zs = []
        for negative_graph in negative_graphs:
            negative_h = negative_graph.ndata['feat'].to(self.device).float()
            negative_x = negative_graph.ndata['coords'].to(self.device).float()
            negative_o = negative_graph.ndata['ori'].to(self.device).float()
            negative_e = negative_graph.edata['feat'].to(self.device).float()
            negative_h = self.feat_model(negative_h)

            src_index, dst_index = negative_graph.edges()
            src_index = src_index.squeeze()
            dst_index = dst_index.squeeze()
            negative_edge_index = torch.stack([src_index, dst_index], dim=0)
            negative_edge_vectors = negative_x[negative_edge_index[0]] - negative_x[
                negative_edge_index[1]]  # 形状为 [n_edges, 3]
            negative_edge_vectors_normalized = _normalize(negative_edge_vectors).unsqueeze(-2)  # 变为 [n_edges, 1, 3]
            negative_edge_inputs = (negative_e, negative_edge_vectors_normalized)
            negative_edge_inputs = self.W_e(negative_edge_inputs)
            negative_node_inputs = self.W_v(negative_h)
            tensor1, tensor2 = negative_node_inputs
            tensor2 = tensor2 + negative_o  
            negative_node_inputs = (tensor1, tensor2)
            for layer in self.gvp:
                negative_node_inputs = layer(negative_node_inputs, negative_edge_index, negative_edge_inputs)
            negative = self.W_out(negative_node_inputs)
            negative_z = self.dense(negative)
            negative_zs.append(negative_z)

        # anchor graph pooling
        anchor_graph.ndata['feat'] = anchor_z
        anchor_zg = dgl.mean_nodes(anchor_graph, 'feat')
        # positive graph pooling
        positive_graph.ndata['feat'] = positive_z
        positive_zg = dgl.mean_nodes(positive_graph, 'feat')
        # negatives graph pooling
        negative_zgs = []
        for negative_graph, negative_z in zip(negative_graphs, negative_zs):
            negative_graph.ndata['feat'] = negative_z
            negative_zg = dgl.mean_nodes(negative_graph, 'feat')
            negative_zgs.append(negative_zg)

        # contrast loss
        contrast_loss = self.nt_xent_loss(anchor_zg, positive_zg, negative_zgs,
                                          anchor_score, neg_scores, anchor_target, neg_targets)

        # score return
        x, y_level = self.mlp_readout(anchor_zg) 
        pred_lddt = x
        pred_lddt_level = y_level

        return pred_lddt, pred_lddt_level, contrast_loss

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
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                 step_size=_lr_reduce_factor,
                                                                 gamma=0.5),
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
                    'scheduler': torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                                 step_size=_lr_reduce_factor, gamma=0.5),
                    'monitor': 'val_loss'
                }
            }

    def training_step(self, train_batch, batch_idx):

        # input
        anchor_graph, lddt_scores, lddt_level, dockq_scores, anchor_target, \
        positive_graph, negative_graphs, neg_dockq_scores, neg_targets = train_batch

        # batch information
        batch_size = anchor_graph.batch_size  # batchsize

        # model forward 
        pred_lddt, pred_lddt_level, train_contrast_loss = self.forward(anchor_graph, positive_graph, negative_graphs,
                                                                       dockq_scores, neg_dockq_scores,
                                                                       anchor_target, neg_targets)

        # lddt loss
        train_mse = self.criterion(pred_lddt.view(-1), lddt_scores.view(-1))
        self.log('train_mse', train_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        train_mse *= self.mse_weight

        # ce
        train_ce = F.cross_entropy(pred_lddt_level, lddt_level.squeeze(1))
        self.log('train_ce', train_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        train_ce *= self.ce_weight

        # acc
        train_acc = self.criterion_acc(F.softmax(pred_lddt_level, dim=1), lddt_level.squeeze(1))

        # cl loss
        train_contrast_loss *= self.cl_weight

        # total loss
        total_loss = train_mse + train_ce + train_contrast_loss
        train_loss = total_loss

        # log training metrics
        self.train_losses.append(train_loss.item())
        self.train_mse.append(train_mse.item())
        self.train_ce.append(train_ce.item())
        self.train_cl_loss.append(train_contrast_loss.item())
        self.train_acc.append(train_acc.item())

        # log
        self.log('train_cl_loss', train_contrast_loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=_batch_size)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        wandb.log({
            'train_loss': train_loss.item(),
            'train_mse': train_mse.item(),
            'train_ce': train_ce.item(),
            'train_cl_loss': train_contrast_loss.item(),
            'train_acc': train_acc.item(),
            'epoch': self.current_epoch
        })

        return train_loss

    def validation_step(self, val_batch, batch_idx):

        # input
        anchor_graph, lddt_scores, lddt_level, dockq_scores, anchor_target, \
        positive_graph, negative_graphs, neg_dockq_scores, neg_targets = val_batch
        # print(f"val_batch: {val_batch}")

        # batch information
        batch_size = anchor_graph.batch_size  # batchsize

        # model forward
        pred_lddt, pred_lddt_level, val_contrast_loss = self.forward(anchor_graph, positive_graph, negative_graphs,
                                                                     dockq_scores, neg_dockq_scores,
                                                                     anchor_target, neg_targets)

        # lddt loss
        val_mse = self.criterion(pred_lddt.view(-1), lddt_scores.view(-1))
        self.log('val_mse', val_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        val_mse *= self.mse_weight

        # ce
        val_ce = F.cross_entropy(pred_lddt_level, lddt_level.squeeze(1))
        self.log('val_ce', val_ce, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        val_ce *= self.ce_weight

        # acc
        val_acc = self.criterion_acc(F.softmax(pred_lddt_level, dim=1), lddt_level.squeeze(1))

        # cl loss
        val_contrast_loss *= self.cl_weight

        # total loss
        total_loss = val_mse + val_ce + val_contrast_loss

        val_loss = total_loss

        # log val metrics
        self.val_losses.append(val_loss.item())
        self.val_mse.append(val_mse.item())
        self.val_ce.append(val_ce.item())
        self.val_cl_loss.append(val_contrast_loss.item())
        self.val_acc.append(val_acc.item())

        # log
        self.log('val_cl_loss', val_contrast_loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=_batch_size)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=_batch_size)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        wandb.log({
            'val_loss': val_loss.item(),
            'val_mse': val_mse.item(),
            'val_ce': val_ce.item(),
            'val_cl_loss': val_contrast_loss.item(),
            'val_acc': val_acc.item(),
            'epoch': self.current_epoch
        })

        return val_loss

    def on_train_end(self):
        print("Training is finished. Saving losses...")
        # save losses json file
        save_path = r"../data/temp_out/losses/losses_gvp_gcl_lddt_v1.json"

        losses_dict = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_mse": self.train_mse,
            "train_ce": self.train_ce,
            "train_cl_loss": self.train_cl_loss,
            "train_acc": self.train_acc,
            "val_mse": self.val_mse,
            "val_ce": self.val_ce,
            "val_cl_loss": self.val_cl_loss,
            "val_acc": self.val_acc
        }

        with open(save_path, 'w') as f:
            json.dump(losses_dict, f, indent=2) 

        print(f"training and validation losses are saved to {save_path}")


def collate_fn(batch):
    """
    defined collate_fn，process batch data including anchor, positive and negatives.
    """
    # define lDDT level mapping
    level_mapping = {
        "very low": 0,
        "low": 1,
        "confident": 2,
        "very high": 3
    }

    # batch data
    (graphs, lddt_scores, lddt_level, dockq_scores, target_name,
     positive_graph, pos_dockq_score, pos_target,
     negative_graphs, neg_dockq_scores, neg_targets) = zip(*batch)

    lddt_level_int = [level_mapping[level] for level in lddt_level]
    lddt_level_int = [int(i) for i in lddt_level_int]  

    # Step 2: process anchor
    batched_graph = dgl.batch(graphs)
    batched_lddt_scores = torch.tensor(np.array(lddt_scores)).unsqueeze(1)
    batched_dockq_scores = torch.tensor(np.array(dockq_scores)).unsqueeze(1)  

    # Step 3: process positive
    batched_positive_graph = dgl.batch(positive_graph)
    batched_pos_dockq_score = torch.tensor(np.array(pos_dockq_score)).unsqueeze(1)  

    # Step 4: process negatives
    batched_negative_graphs = [neg_graph for neg_list in negative_graphs for neg_graph in neg_list]
    batched_neg_dockq_scores = [score for neg_list in neg_dockq_scores for score in neg_list]  
    batched_neg_targets = [tgt for neg_list in neg_targets for tgt in neg_list] 

    # Step 5: transform lDDT level to tensor
    batched_lddt_level = torch.tensor(lddt_level_int, dtype=torch.long).unsqueeze(1)

    # return batch data
    return (
        batched_graph, batched_lddt_scores, batched_lddt_level, batched_dockq_scores, target_name,  # anchor
        batched_positive_graph,  # positive
        batched_negative_graphs, batched_neg_dockq_scores, batched_neg_targets  # negatives
    )


def load_dataset_from_txt(trainset_path, valset_path):
    # read training dataset
    with open(trainset_path, 'r') as train_f:
        train_files = [line.strip() for line in train_f.readlines()]

    # read val dataset
    with open(valset_path, 'r') as val_f:
        val_files = [line.strip() for line in val_f.readlines()]

    return train_files, val_files


data_folder = r'path/to/DGL'  # replace your dataset path
lddt_csv = r'path/to/lddt.csv'
dockq_csv = r'path/to/dockq.csv'

train_files, val_files = load_dataset_from_txt('path/to/trainset.txt',
                                               'path/to/valset.txt')

train_dataset = GCLDGLDataset(data_folder, lddt_csv, dockq_csv, file_list=train_files)
val_dataset = GCLDGLDataset(data_folder, lddt_csv, dockq_csv, file_list=val_files)

train_loader = DataLoader(train_dataset, batch_size=_batch_size, num_workers=4, persistent_workers=True,
                          shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=_batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)

model = QAModel().to('cuda')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


model.apply(init_weights)


# logger
wandb_logger = WandbLogger(project="QA",
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
    filename='{epoch}-{val_loss:.5f}',  
    monitor='val_loss',  
    save_top_k=3, 
    mode='min', 
    save_weights_only=True, 
    verbose=True  
)

saw = StochasticWeightAveraging(swa_epoch_start=0.7,
                                swa_lrs=_init_lr * 0.1,
                                annealing_epochs=10,
                                annealing_strategy='cos')

# define a trainer
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,  
    num_nodes=1,
    max_epochs=_epochs,
    logger=wandb_logger,  
    callbacks=[early_stop_callback, checkpoint_callback, lr_monitor, saw],
    sync_batchnorm=True,
    accumulate_grad_batches=_accumulate_grad_batches
)

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)

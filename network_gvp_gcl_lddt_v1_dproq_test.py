import json
import os
from datetime import datetime

import dgl
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from model.dataset import MLPReadoutlddtClass, MLPReadoutlddtClassV2, MLPReadoutlddtClassV3
# 定义参数
from model.GVP_GNN import GVPConvLayer, _normalize, LayerNorm, GVP

CUDA_LAUNCH_BLOCKING = 1

# load config file
config_file = f'/mnt/d/pycharm/pycharmProjects/ideamodel3qa/model/config/config_gvp_gcl_v1.json'
with open(config_file) as f:
    print(f'Loading config file {config_file}')
    config = json.load(f)

# 使用参数
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

print(f"Training with dataset: {_dataset}, learning rate: {_init_lr}")

# 设置日志和检查点保存目录
_wb_out_dir = './logs/GVP_GCL_v1'
_ckpt_out_dir = './checkpoints/'

# 确保输出目录存在
os.makedirs(_wb_out_dir, exist_ok=True)
os.makedirs(_ckpt_out_dir, exist_ok=True)

# 设定当前时间
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

        # GVPConvLayer 参数
        self.gvp = nn.ModuleList(GVPConvLayer(
            node_dims=self.node_hidden_dim,  # 节点特征维度
            edge_dims=(self.edge_scalar_feature_dim, self.edge_vector_feature_dim),  # 边特征维度
            activations=(F.relu, None),  # 激活函数（标量，矢量）
            vector_gate=True  # 是否使用矢量门控
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

        self.mlp_readout = MLPReadoutlddtClassV3(input_dim=self.node_out_dim, output_dim=1, dp_rate=_mlp_dp_rate)

    def nt_xent_loss(self, anchor, positive, negatives, anchor_score, neg_scores, anchor_target, neg_targets,
                     temperature=0.5):
        """
        计算基于标签权重的 NT-Xent 对比损失。

        :param anchor: 主样本嵌入
        :param positive: 正样本嵌入
        :param negatives: 负样本嵌入 (list)
        :param neg_scores: 负样本对应的 DockQ 分数 (list)
        :param anchor_score: 主样本的 DockQ 分数
        :param neg_targets: 负样本对应的 target 名称 (list)
        :param anchor_target: 主样本的 target 名称
        :param temperature: 温度参数
        :return: 计算出的对比损失
        """

        # 计算主样本与正样本的相似度
        pos_similarity = torch.exp(torch.cosine_similarity(anchor, positive) / temperature)

        neg_similarities = []
        neg_weights = []

        for negative, neg_score, neg_target in zip(negatives, neg_scores, neg_targets):
            # 计算负样本相似度
            neg_similarity = torch.exp(torch.cosine_similarity(anchor, negative) / temperature)
            neg_similarities.append(neg_similarity)

            # 计算权重：同一 target 下用分数差值，不同 target 设为 1
            weight = torch.abs(anchor_score - neg_score) if neg_target == anchor_target else torch.ones_like(
                anchor_score)
            weight = weight.squeeze()
            neg_weights.append(weight)

        # 堆叠所有负样本相似度 & 权重
        neg_similarities = torch.stack(neg_similarities)  # (num_negatives,)
        neg_weights = torch.stack(neg_weights)  # (num_negatives,)

        # 计算加权负样本相似度之和
        weighted_neg_sum = (neg_similarities * neg_weights).sum(dim=0)

        # 计算损失
        loss = -torch.log(pos_similarity / (pos_similarity + weighted_neg_sum)).mean()

        return loss

    def forward(self, anchor_graph, positive_graph=None, negative_graphs=None, anchor_score=None, neg_scores=None,
                anchor_target=None, neg_targets=None, scatter_mean=True, dense=True):

        # 主样本嵌入
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
        anchor_edge_inputs = self.W_e(anchor_edge_inputs)
        anchor_node_inputs = self.W_v(anchor_h)

        # 添加残差连接
        tensor1, tensor2 = anchor_node_inputs
        tensor2 = tensor2 + anchor_o  # 将 ori 加到 tensor2 上

        anchor_node_inputs = (tensor1, tensor2)

        for i, layer in enumerate(self.gvp):
            anchor_node_inputs = layer(anchor_node_inputs, anchor_edge_index, anchor_edge_inputs)

        anchor = self.W_out(anchor_node_inputs)
        anchor_z = self.dense(anchor)

        # 主样本图级嵌入
        anchor_graph.ndata['feat'] = anchor_z
        anchor_final_zg = dgl.mean_nodes(anchor_graph, 'feat')

        # 通过MLP层（多层感知机）进行最终预测
        x, y_level = self.mlp_readout(anchor_final_zg)  # 期望 pred_lddt 形状为 [batch_size, 1]
        pred_lddt = x
        pred_lddt_level = y_level

        # 如果正负样本图存在，则计算对比损失（如果对比学习还是需要）
        if positive_graph is not None and negative_graphs is not None:
            # 正样本嵌入
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
            # 添加残差连接
            tensor1, tensor2 = positive_node_inputs
            tensor2 = tensor2 + positive_o  # 将 ori 加到 tensor2 上
            positive_node_inputs = (tensor1, tensor2)
            for layer in self.gvp:
                positive_node_inputs = layer(positive_node_inputs, positive_edge_index, positive_edge_inputs)
            positive = self.W_out(positive_node_inputs)
            positive_z = self.dense(positive)

            # 负样本嵌入
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
                # 添加残差连接
                tensor1, tensor2 = negative_node_inputs
                tensor2 = tensor2 + negative_o  # 将 x 加到 tensor2 上
                negative_node_inputs = (tensor1, tensor2)
                for layer in self.gvp:
                    negative_node_inputs = layer(negative_node_inputs, negative_edge_index, negative_edge_inputs)
                negative = self.W_out(negative_node_inputs)
                negative_z = self.dense(negative)
                negative_zs.append(negative_z)

            # 主样本图级嵌入
            anchor_graph.ndata['feat'] = anchor_z
            anchor_zg = dgl.mean_nodes(anchor_graph, 'feat')

            # 正样本图级嵌入
            positive_graph.ndata['feat'] = positive_z
            positive_zg = dgl.mean_nodes(positive_graph, 'feat')

            # 负样本图级嵌入
            negative_zgs = []
            for negative_graph, negative_z in zip(negative_graphs, negative_zs):
                negative_graph.ndata['feat'] = negative_z
                negative_zg = dgl.mean_nodes(negative_graph, 'feat')
                negative_zgs.append(negative_zg)

            # 计算对比损失
            contrast_loss = self.nt_xent_loss(anchor_zg, positive_zg, negative_zgs,
                                              anchor_score, neg_scores, anchor_target, neg_targets)
            return pred_lddt, pred_lddt_level, contrast_loss

        return pred_lddt, pred_lddt_level  # 如果没有正负样本图，只返回预测

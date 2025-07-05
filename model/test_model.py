import json
import dgl
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from model.dataset import MLPReadoutlddtClass
from model.GVP_GNN import GVPConvLayer, _normalize, LayerNorm, GVP

CUDA_LAUNCH_BLOCKING = 1

# load config file
config_file = f'path/to/pre_train_knn10_seed42.json'
with open(config_file) as f:
    print(f'Loading config file {config_file}')
    config = json.load(f)

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

        pl.seed_everything(_seed)

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

        # GVPConvLayer 
        self.gvp = nn.ModuleList(GVPConvLayer(
            node_dims=self.node_hidden_dim,  
            edge_dims=(self.edge_scalar_feature_dim, self.edge_vector_feature_dim),  
            activations=(F.relu, None),  
            vector_gate=True 
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

    def nt_xent_loss(self, anchor, positive, negatives, anchor_score, neg_scores, anchor_target, neg_targets,
                     temperature=0.07):

        pos_similarity = torch.exp(torch.cosine_similarity(anchor, positive) / temperature)

        neg_similarities = []
        neg_weights = []

        for negative, neg_score, neg_target in zip(negatives, neg_scores, neg_targets):
            neg_similarity = torch.exp(torch.cosine_similarity(anchor, negative) / temperature)
            neg_similarities.append(neg_similarity)

            weight = torch.abs(anchor_score - neg_score) if neg_target == anchor_target else torch.ones_like(
                anchor_score)
            weight = weight.squeeze()
            neg_weights.append(weight)

        neg_similarities = torch.stack(neg_similarities)  # (num_negatives,)
        neg_weights = torch.stack(neg_weights)  # (num_negatives,)

        weighted_neg_sum = (neg_similarities * neg_weights).sum(dim=0)

        loss = -torch.log(pos_similarity / (pos_similarity + weighted_neg_sum)).mean()

        return loss

    def forward(self, anchor_graph, positive_graph=None, negative_graphs=None, anchor_score=None, neg_scores=None,
                anchor_target=None, neg_targets=None, scatter_mean=True, dense=True):

        anchor_h = anchor_graph.ndata['feat'].to(self.device).float()
        anchor_x = anchor_graph.ndata['coords'].to(self.device).float()
        anchor_o = anchor_graph.ndata['ori'].to(self.device).float()
        anchor_e = anchor_graph.edata['feat'].to(self.device).float()
        anchor_h = self.feat_model(anchor_h)

        src_index, dst_index = anchor_graph.edges()
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()
        anchor_edge_index = torch.stack([src_index, dst_index], dim=0)
        anchor_edge_vectors = anchor_x[anchor_edge_index[0]] - anchor_x[anchor_edge_index[1]] 
        anchor_edge_vectors_normalized = _normalize(anchor_edge_vectors).unsqueeze(-2) 
        anchor_edge_inputs = (anchor_e, anchor_edge_vectors_normalized)
        anchor_edge_inputs = self.W_e(anchor_edge_inputs)
        anchor_node_inputs = self.W_v(anchor_h)

        tensor1, tensor2 = anchor_node_inputs
        tensor2 = tensor2 + anchor_o 

        anchor_node_inputs = (tensor1, tensor2)

        for i, layer in enumerate(self.gvp):
            anchor_node_inputs = layer(anchor_node_inputs, anchor_edge_index, anchor_edge_inputs)

        anchor = self.W_out(anchor_node_inputs)
        anchor_z = self.dense(anchor)

        anchor_graph.ndata['feat'] = anchor_z
        anchor_final_zg = dgl.mean_nodes(anchor_graph, 'feat')

        x, y_level = self.mlp_readout(anchor_final_zg)  
        pred_lddt = x
        pred_lddt_level = y_level

        if positive_graph is not None and negative_graphs is not None:
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
            positive_edge_vectors_normalized = _normalize(positive_edge_vectors).unsqueeze(-2)  
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
                negative_edge_vectors_normalized = _normalize(negative_edge_vectors).unsqueeze(-2)  
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

            anchor_graph.ndata['feat'] = anchor_z
            anchor_zg = dgl.mean_nodes(anchor_graph, 'feat')

            positive_graph.ndata['feat'] = positive_z
            positive_zg = dgl.mean_nodes(positive_graph, 'feat')

            negative_zgs = []
            for negative_graph, negative_z in zip(negative_graphs, negative_zs):
                negative_graph.ndata['feat'] = negative_z
                negative_zg = dgl.mean_nodes(negative_graph, 'feat')
                negative_zgs.append(negative_zg)

            contrast_loss = self.nt_xent_loss(anchor_zg, positive_zg, negative_zgs,
                                              anchor_score, neg_scores, anchor_target, neg_targets)
            return pred_lddt, pred_lddt_level, contrast_loss

        return pred_lddt, pred_lddt_level

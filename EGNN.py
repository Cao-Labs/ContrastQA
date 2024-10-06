##################################################################################################################
# Obtained from https://github.com/vgsatorras/egnn/blob/main/models/egnn_clean/egnn_clean.py
# Credit: 'E(n) Equivariant Graph Neural Networks' by Victor Garcia Satorras, Emiel Hogeboom, Max Welling
#          Proceedings of the 38th International Conference on Machine Learning, PMLR 139:9323-9332, 2021.
##################################################################################################################

from torch import nn
import torch
from torch.nn import init


# 实现了一种等变卷积操作，通过节点、边和坐标特征的结合，实现了对节点特征以及坐标的更新（边特征如果有变化可能会更新）
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """
    # input_nf:输入节点特征的维度
    # output_nf:输出节点特征的维度
    # hidden_nf：隐藏层特征维度。通常这个维度设置得比输入和输出特征维度大，以便能够捕捉更复杂的特征
    # edges_in_d：输入边特征的维度
    # act_fn:激活函数
    # residual：是否使用残差连接。设置为 True 时，输入节点特征会与输出特征相加，这有助于缓解深层网络中的梯度消失问题。
    # attention：是否启用注意力机制。如果设置为 True，则在边特征计算中将引入一个额外的注意力权重，以增强模型对重要边的关注。
    # normalize：是否对坐标消息进行归一化。当设置为 True 时，计算出的坐标差异将被其模长归一化，有助于稳定模型训练。
    # coords_agg：坐标聚合的方式。可以选择 'sum' 或 'mean'，用于决定在更新节点坐标时如何聚合来自不同边的信息。
    # tanh：是否在坐标模型的输出中使用Tanh激活函数。设置为 True 将会限制输出范围，有助于模型的稳定性，但可能会降低准确性。
    
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True,
                 attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        # 定义一个用于计算边特征的多层感知机（MLP），接收源节点和目标节点特征、径向距离和边特征。
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # 定义一个用于计算节点特征的MLP，将边特征的聚合结果与节点特征结合
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # 定义坐标特征的MLP，包括激活函数和可选的tanh激活函数
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # 如果启用注意力机制，则定义相应的MLP。
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    # 计算边特征，结合源节点、目标节点、径向距离和可选的边特征。
    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        # 如果启用注意力机制，使用注意力权重调整输出。
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    # 计算节点特征，通过边索引聚合边特征。out:节点特征，agg：聚合后的边特征
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        # 如果启用残差连接，将原始节点特征加到输出上。
        if self.residual:
            out = x + out
        return out, agg

    # 计算坐标特征的聚合，基于边的索引和特征进行处理
    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    # 计算节点之间的距离和差异，得到径向特征
    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        # 可选地对坐标差异进行归一化处理。
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    # 定义前向传播逻辑，依次调用边特征、坐标特征和节点特征的计算方法
    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


# 实现了一种等变图神经网络的结构，通过多个 E_GCL 层的叠加来处理图数据，
# 使得模型能够有效地捕捉节点间的关系和几何信息，最终输出更新的节点特征和坐标
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(),
                 n_layers=4, residual=True, attention=False, normalize=False, tanh=False):
        """

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        """

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # 定义线性层
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)

        # 对 embedding_in 和 embedding_out 进行 Xavier 初始化
        init.xavier_uniform_(self.embedding_in.weight)
        init.xavier_uniform_(self.embedding_out.weight)

        if self.embedding_in.bias is not None:
            init.zeros_(self.embedding_in.bias)
        if self.embedding_out.bias is not None:
            init.zeros_(self.embedding_out.bias)

        # 创建 n_layers 个 E_GCL 层
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                                                attention=attention, normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("Input h contains nan or inf values.")

        print(f"before embed h shape: {h.shape}, h values: {h}")
        h = self.embedding_in(h)
        print(f"after embedding h shape: {h.shape}, h values: {h}")
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
            print(f"Layer {i}, h shape: {h.shape}, h values: {h}")

        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    # rows 和 cols，用于存储边的起始节点和目标节点的索引。
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    # 将 rows 和 cols 列表组合成一个列表 edges，表示所有边的起始节点和目标节点。
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    # 创建一个边特征的张量 edge_attr，其大小为边的数量乘以 batch_size，每个边特征的值为 1
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    # 将 rows 和 cols 转换为长整型的张量。
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    # 如果批量大小为 1，直接返回边信息和边特征
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        # 将 rows 和 cols 列表合并成一个新的边列表，形成批量的边索引
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    # Run EGNN
    h, x = egnn(h, x, edges, edge_attr)

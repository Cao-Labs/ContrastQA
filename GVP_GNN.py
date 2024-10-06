import functools

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import functional as F
from torch_scatter import scatter_add


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


# 按元素对任意数量的元组(s, V)求和
def tuple_sum(*args):
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))


# 按元素顺序在指定的维度上连接任意数量的元组(s, V)。
def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


# 根据索引 idx 从元组 x 中提取元素,返回提取后的标量部分和向量部分
def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


# 计算张量 x 的 L2 范数，确保结果不小于最小值 eps
# axis 指定计算的轴，keepdims 控制维度保持，sqrt 指定是否返回平方根
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


# 将合并的张量 x 拆分回元组形式
def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


# 将元组 (s, V) 合并为一个张量，向量通道展平并附加到标量通道后面
def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


# in_dims：输入维度
# out_dims：输出维度
# h_dim：中间层矢量通道的数量，可选
# activations：激活函数元组（标量激活函数，矢量激活函数）
# vector_gate：矢量门控参数，是否使用矢量门控。(如果' True '， vector_act将在矢量门控中用作sigma^+)
class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        # 如果输入中有向量维度：
        # 设置隐层维度，默认为 h_dim 或输入和输出向量维度的最大值。
        # 定义线性层 wh 用于从输入向量到隐层向量的映射。
        # 定义线性层 ws 用于将隐层向量和输入标量拼接后映射到输出标量。
        # 如果有输出向量维度，则定义线性层 wv 用于从隐层向量到输出向量的映射。如果使用向量门控，则定义 wsv。
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            # print(f'self.vi={self.vi}')
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        # 如果没有输入向量，则只定义标量的线性层 ws。
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        # 定义一个空的参数，用于确定设备（如 CPU 或 GPU）。
        self.dummy_param = nn.Parameter(torch.empty(0))

    # 前向传播方法接受输入 x，可以是包含标量和向量的元组，或仅为标量的张量。
    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            # print(f"before Adjusted shape of v: {v.shape}")
            v = torch.transpose(v, -1, -2)

            # print(f"Input shape to linear layer: {v.shape}")
            vh = self.wh(v)

            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                # 如果使用向量门控：计算门控值 gate，将输出向量 v 乘以门控值的 sigmoid
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


# 实现了矢量通道的 dropout
class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each vector channel are dropped together.
    """

    # drop_rate：表示丢弃的比例
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    # x：tensor，表示向量通道
    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        # 获取当前设备
        device = self.dummy_param.device
        # 如果不在训练模式下，直接返回输入张量 x，不进行丢弃
        if not self.training:
            return x
        # 创建掩码,应用掩码并缩放
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


# 用于处理标量和矢量的组合dropout
class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    # 分别创建标量通道和矢量通道的dropout层
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    # 接受输入 x，可以是包含标量和向量的元组，或仅为标量的张量
    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        # 如果输入是单个张量，直接通过 sdropout 进行标量 dropout 处理。
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        # 如果输入是元组，解包为标量 s 和向量 v，分别应用 sdropout 和 vdropout，返回结果
        s, v = x
        return self.sdropout(s), self.vdropout(v)


# 对元组 (s, V) 即标量和矢量进行组合的层归一化
class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    # x，可以是包含标量和向量的元组，或仅为标量的张量
    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        # 如果向量维度 self.v 为零，表示没有向量通道，直接对输入张量 x 进行标量归一化并返回结果。
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        # 计算向量归一化
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        # 返回标量归一化的结果和向量归一化后的结果
        return self.scalar_norm(s), v / vn


# 用于图卷积或信息传播，使用几何向量感知器（GVP）进行处理，输入图的节点和边嵌入，并返回新的节点嵌入
class GVPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        # 创建一个新的 GVP 类，该类的激活函数和向量门控参数预设为当前实例的参数
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)

        # 构建消息函数模块列表
        module_list = module_list or []
        # 如果 module_list 仍为空且层数为 1，添加一个 GVP 模块到列表中，输入维度是节点和边的总和，输出维度是目标维度。
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            # 如果层数大于 1，首先添加一个 GVP 模块，其输入维度与前面相同，输出维度为目标维度
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                # 循环添加剩余的 GVP 模块，确保每一层的输入输出维度一致。
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                # 最后添加一个 GVP 模块，不使用激活函数。
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        # 将构建好的模块列表转化为一个顺序容器，方便后续调用
        self.message_func = nn.Sequential(*module_list)

    # 输入 x（节点嵌入）、edge_index（边索引）和 edge_attr（边特征）
    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """

        x_s, x_v = x
        # print(f"x_s shape: {x_s.shape}")
        # print(f"x_v shape before reshape: {x_v.shape}")

        # 调用 propagate 方法进行信息传播，传递边索引、节点特征和边特征
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    # 处理节点和边的特征
    def message(self, s_i, v_i, s_j, v_j, edge_attr):

        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)

        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


# 执行图卷积和消息传递，同时进行节点嵌入的残差更新
# node_dims:输入节点嵌入维度（标量特征，矢量特征）
# edge_dims：输入边嵌入维度（标量特征，矢量特征）
# n_message：用于消息函数中GVP的数量
# n_feedforward：用于前馈函数中GVP的数量
# drop_rate：所有dropout层drop概率
# autoregressive：如果' True '，这个' GVPConvLayer '将与src >= dst的消息的不同输入节点嵌入集一起使用
# activations：用于GVP中的激活函数元组（标量激活，矢量激活）
# vector_gate：是否使用矢量门控
class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.
    使用几何向量感知器的全图卷积/消息传递层。残差用聚合的传入消息更新节点嵌入，对节点嵌入应用点向前馈网络，并返回更新后的节点嵌入。
    To only compute the aggregated messages, see `GVPConv`.
    要只计算聚合的消息，请参见' GVPConv '。
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):

        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        # 对节点嵌入进行层归一化
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        # 初始化前馈网络的模块列表。如果前馈网络层数为 1，则添加一个 GVP 模块，不使用激活函数
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    # 输入参数包括节点嵌入、边索引、边特征、自回归节点嵌入和节点掩码
    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        # 解包边索引，创建一个掩码，筛选出源节点索引小于目标节点索引的边
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            # 计算前向和后向边的消息，聚合得到更新的节点特征 dh
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            # 根据接收到的消息数量对更新的特征进行归一化
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        # 如果没有提供自回归节点嵌入，直接使用当前节点嵌入计算更新特征
        else:
            dh = self.conv(x, edge_index, edge_attr)

        # 如果提供节点掩码，则只更新被掩码的节点
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        # 更新节点嵌入
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        # 如果使用了节点掩码，更新原始节点嵌入中的对应节点，确保只有被掩码的节点被更新
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

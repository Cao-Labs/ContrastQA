import os
import glob
import pandas as pd
import dgl
import torch
from torch import nn
from torch.utils.data import Dataset
import io


class MLPReadoutClass(nn.Module):
    """Read-out Module"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5, L=2):
        super(MLPReadoutClass, self).__init__()
        self.L = L
        self.list_FC_layer = nn.Sequential()

        # 确保 input_dim 足够大
        if input_dim < 2 ** L:
            raise ValueError(f"input_dim ({input_dim}) 应该至少是 2^L ({2 ** L})")

        for i in range(L):
            self.list_FC_layer.add_module(
                f'Linear {i}',
                nn.Linear(input_dim // 2 ** i, input_dim // 2 ** (i + 1), bias=True)
            )
            self.list_FC_layer.add_module(
                f'BN {i}',
                nn.BatchNorm1d(input_dim // 2 ** (i + 1))
            )
            self.list_FC_layer.add_module(
                f'relu {i}',
                nn.LeakyReLU()
            )
            self.list_FC_layer.add_module(
                f'dp {i}',
                nn.Dropout(p=dp_rate)
            )

        self.last_layer_classification = nn.Linear(input_dim // 2 ** L, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_2 = self.last_layer_classification(x)  # class label
        pred_dockq = torch.sigmoid(self.last_layer(y_2))  # dockq_score
        return pred_dockq, y_2


class MLPReadoutlddtClass(nn.Module):
    """Read-out Module"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5, L=2):
        super(MLPReadoutlddtClass, self).__init__()
        self.L = L
        self.list_FC_layer = nn.Sequential()

        # 确保 input_dim 足够大
        if input_dim < 2 ** L:
            raise ValueError(f"input_dim ({input_dim}) 应该至少是 2^L ({2 ** L})")

        for i in range(L):
            self.list_FC_layer.add_module(
                f'Linear {i}',
                nn.Linear(input_dim // 2 ** i, input_dim // 2 ** (i + 1), bias=True)
            )
            self.list_FC_layer.add_module(
                f'BN {i}',
                nn.BatchNorm1d(input_dim // 2 ** (i + 1))
            )
            self.list_FC_layer.add_module(
                f'relu {i}',
                nn.LeakyReLU()
            )
            self.list_FC_layer.add_module(
                f'dp {i}',
                nn.Dropout(p=dp_rate)
            )

        self.last_layer_classification = nn.Linear(input_dim // 2 ** L, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_2 = self.last_layer_classification(x)

        # 计算局部 LDDT 分数
        node_lddt_scores = torch.sigmoid(self.last_layer(y_2))  # 输出范围在 [0, 1] 内

        return node_lddt_scores


class TestData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str):
        # 保存 DGL 文件名列表
        self.file_names = os.listdir(dgl_folder)
        self.data_list = [os.path.join(dgl_folder, i) for i in self.file_names]
        self.data = []
        self._prepare()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _prepare(self):
        for i in range(len(self.data_list)):
            g, tmp = dgl.data.utils.load_graphs(self.data_list[i])
            self.data.append(g[0])


def collate(samples) -> dgl.DGLGraph:
    """Customer collate function"""
    batched_graph = dgl.batch(samples)
    return batched_graph


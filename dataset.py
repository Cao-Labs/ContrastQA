import os
import glob
import pandas as pd
import dgl
import torch
from torch import nn
from torch.utils.data import Dataset
import io


# class DGLDataset(Dataset):
#     def __init__(self, data_folder, score_csv):
#         self.data_folder = data_folder
#         self.score_data = self.load_score_data(score_csv)
#         self.dgl_files = sorted(glob.glob(os.path.join(data_folder, '*.dgl')))
#
#     def __len__(self):
#         return len(self.dgl_files)
#
#     def __getitem__(self, idx):
#         dgl_file = self.dgl_files[idx]
#         dgl_graphs, _ = dgl.load_graphs(dgl_file)
#
#         if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
#             dgl_graph = dgl_graphs[0]
#         else:
#             raise ValueError(f"No DGL graph loaded from {dgl_file}")
#
#         # Print node and edge features for debugging
#         # print(f"Graph ID: {idx}")
#         # print(f"Node Features: {dgl_graph.ndata['feat']}")
#         # print(f"Edge Features: {dgl_graph.edata['weight']}")
#
#         score = self.get_score(idx + 1)  # Assuming idx is a global index
#
#         return dgl_graph, score
#
#     def load_score_data(self, score_csv):
#         with open(score_csv, 'r') as f:
#             content = f.read()
#
#         parts = content.split("=== Contents of ")
#         score_data = {}
#         global_index = 1
#
#         for part in parts[1:]:  # Skip the first part as it doesn't contain useful data
#             lines = part.strip().split('\n')
#             file_name = lines[0].split('.csv')[0]
#             csv_content = '\n'.join(
#                 [line for line in lines[1:] if 'NULL' not in line and line.strip()])  # Filter out lines containing 'NULL' and empty lines
#
#             # Read csv content and skip irrelevant lines
#             csv_io = io.StringIO(csv_content)
#             df = pd.read_csv(csv_io)
#
#             # Check if required columns exist
#             if 'Decoy Index' not in df.columns or 'Best DockQ' not in df.columns:
#                 raise KeyError(f"Required columns are missing in {file_name}")
#
#             # Convert each Decoy Index in the file to global index
#             for local_index in df['Decoy Index']:
#                 score_data[global_index] = df.loc[df['Decoy Index'] == local_index, 'Best DockQ'].values[0]
#                 global_index += 1
#
#         return score_data
#
#     def get_score(self, decoy_index):
#         if decoy_index in self.score_data:
#             return self.score_data[decoy_index]
#         else:
#             raise ValueError(f"No score found for Decoy Index {decoy_index}")
#
#
# # 示例使用
# data_folder = r'D:\pycharm\pycharmProjects\ideamodel2qa\ideamodel2qa\data\temp_out\DG_1_DGL'
# score_csv = r'D:\pycharm\pycharmProjects\ideamodel2qa\ideamodel2qa\data\data\train\Final_dockq.csv'
# dataset = DGLDataset(data_folder, score_csv)
#
# for i in range(len(dataset)):
#     try:
#         dgl_graph, score = dataset[i]
#     except ValueError as e:
#         print(e)


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

        node_lddt_scores = torch.mean(node_lddt_scores)

        return node_lddt_scores


class TestData(Dataset):
    """Data loader"""
    def __init__(self, dgl_folder: str):
        self.data_list = os.listdir(dgl_folder)
        self.data_list = [os.path.join(dgl_folder, i) for i in self.data_list]
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


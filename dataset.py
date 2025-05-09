import glob
import io
import os
import random
import re
from io import StringIO

import dgl
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset
import torch
import torch.nn as nn


class GCLDGLDataset(Dataset):
    def __init__(self, data_folder, lddt_csv, dockq_csv, file_list=None, batch_size=4):
        """
        初始化 DGLDataset
        :param data_folder: 数据集所在文件夹路径
        :param lddt_csv: 包含 lDDT 分数数据的 CSV 文件路径
        :param dockq_csv: 包含 DockQ 分数和 CAPRI 类别的 CSV 文件路径
        :param file_list: 文件名列表（可选），如果指定则只加载该列表中的文件
        :param batch_size: 每个 batch 的大小
        """
        self.data_folder = data_folder
        self.lddt_data = self.load_lddt_data(lddt_csv)
        self.dockq_data = self.load_dockq_data(dockq_csv)

        if file_list:
            # 如果提供了文件名列表，只加载该列表中的文件
            self.dgl_files = [os.path.join(data_folder, f"{file_name}") for file_name in file_list]
        else:
            # 否则加载文件夹中的所有 .dgl 文件
            self.dgl_files = sorted(glob.glob(os.path.join(data_folder, '*.dgl')))

        self.batch_size = batch_size

    def __len__(self):
        return len(self.dgl_files)  # 返回 batch 数量

    def __getitem__(self, idx):
        """
        根据索引返回当前样本的图、分数、正负样本的 batch 和正负样本的 DGLGraph
        :param idx: 索引
        :return: 当前样本的图、评分、正样本、负样本及其图
        """
        # 获取当前文件的图和评分
        dgl_file = self.dgl_files[idx]
        dgl_graph, lddt_score, lddt_level, input_name, target_name, \
        dockq_score, capri_category = self.load_graph_and_score(dgl_file)

        # 根据当前目标名称创建一个包含正负样本的 batch（名称列表）
        cluster_batch = self.create_batch(target_name, input_name)

        # 分离正样本和负样本（cluster_batch[0] 为当前样本）
        positive_sample = cluster_batch[1]  # 正样本为列表中的第 2 个
        negative_samples = cluster_batch[2:]  # 负样本为其余部分

        # 加载正样本的图
        positive_graph_path = os.path.join(self.data_folder, f"{positive_sample}.dgl")
        positive_graph, _ = dgl.load_graphs(positive_graph_path)
        positive_graph = positive_graph[0]

        # 获取正样本的 DockQ 分数 & target_name
        pos_dockq_score = self.dockq_data.loc[self.dockq_data['Model'] == positive_sample, 'DockQ'].values[0]
        pos_target = positive_sample.split('_decoy')[0]

        # 处理负样本
        negative_graphs = []
        neg_dockq_scores = []
        neg_targets = []

        for model_name in negative_samples:
            graph_path = os.path.join(self.data_folder, f"{model_name}.dgl")
            graphs, _ = dgl.load_graphs(graph_path)
            negative_graphs.append(graphs[0])

            # 获取负样本 DockQ 分数 & target_name
            neg_score = self.dockq_data.loc[self.dockq_data['Model'] == model_name, 'DockQ'].values[0]
            neg_dockq_scores.append(neg_score)

            neg_target = model_name.split('_decoy')[0]
            neg_targets.append(neg_target)

        return (dgl_graph, lddt_score, lddt_level, dockq_score, target_name,
                positive_graph, pos_dockq_score, pos_target,
                negative_graphs, neg_dockq_scores, neg_targets)

    def load_segmented_data(self, csv_path, score_type="lDDT"):
        """
        加载分段 CSV 文件
        :param csv_path: CSV 文件路径
        :param score_type: 分数类型 ("lDDT" 或其他)
        :return: 包含所有段落数据的 DataFrame
        """
        data_frames = []
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            current_data = []
            for line in lines:
                if line.startswith("==="):  # 新段落的标志
                    if current_data:  # 保存当前段落为 DataFrame
                        data_frames.append(pd.read_csv(StringIO("\n".join(current_data))))
                        current_data = []
                else:
                    current_data.append(line.strip())

            # 添加最后一段
            if current_data:
                data_frames.append(pd.read_csv(io.StringIO("\n".join(current_data))))

        # 合并所有段落
        combined_data = pd.concat(data_frames, ignore_index=True)
        if score_type not in combined_data.columns:
            raise ValueError(f"Expected column '{score_type}' not found in file: {csv_path}")
        return combined_data

    def load_graph_and_score(self, dgl_file):
        """
        加载 DGL 图和评分数据
        :param dgl_file: DGL 文件路径
        :return: 返回图结构、lDDT 评分、目标名称和 CAPRI 类别
        """
        # 加载 DGL 图
        dgl_graphs, _ = dgl.load_graphs(dgl_file)
        if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
            dgl_graph = dgl_graphs[0]
        else:
            raise ValueError(f"No DGL graph loaded from {dgl_file}")

        # 获取文件名并解析目标名称
        input_name = os.path.basename(dgl_file).split('.')[0]
        target_name = input_name.split('_decoy')[0]

        # 提取 decoy 索引
        decoy_index = int(re.search(r'(\d+)$', input_name).group(1))

        # 获取对应文件的评分
        score, lddt_level = self.get_lddt_score(target_name, decoy_index)

        # 获取 DockQ 分数和 CAPRI 类别
        dockq_row = self.dockq_data.loc[self.dockq_data['Model'] == input_name]
        if dockq_row.empty:
            raise ValueError(f"No DockQ data found for model {input_name}")
        dockq_score = dockq_row['DockQ'].values[0]
        capri_category = dockq_row['CAPRI'].values[0]

        return dgl_graph, score, lddt_level, input_name, target_name, dockq_score, capri_category

    def get_lddt_score(self, target_name, decoy_index):
        """
        获取 lDDT 评分和 lDDT level
        :param target_name: 目标名称
        :param decoy_index: decoy 索引
        :return: 对应的 lDDT 评分 和 lDDT level
        """
        if self.lddt_data is None:
            raise ValueError("lDDT data has not been loaded.")

        model_name = f"{target_name}_decoy{decoy_index}"
        if 'Model' not in self.lddt_data.columns:
            raise KeyError("'Model' column is missing from lDDT data.")
        if 'lDDT level' not in self.lddt_data.columns:
            raise KeyError("'lDDT level' column is missing from lDDT data.")

        if model_name in self.lddt_data['Model'].values:
            row = self.lddt_data.loc[self.lddt_data['Model'] == model_name]
            score = row['lDDT'].values[0]
            level = row['lDDT level'].values[0]  # 直接读取 lDDT level

            return score, level
        else:
            raise ValueError(f"No lDDT score found for model {model_name}")

    def load_lddt_data(self, lddt_csv):
        """
        加载 lDDT 分数数据
        :param lddt_csv: 包含 lDDT 分数的 CSV 文件路径
        :return: lDDT 分数数据的 DataFrame
        """
        self.lddt_data = self.load_segmented_data(lddt_csv, score_type="lDDT")
        return self.lddt_data

    def load_dockq_data(self, dockq_csv):
        """
        加载 DockQ 分数和 CAPRI 类别数据
        :param dockq_csv: 包含 DockQ 分数和 CAPRI 类别的 CSV 文件路径
        :return: DockQ 数据的 DataFrame
        """
        self.dockq_data = self.load_segmented_data(dockq_csv, score_type="DockQ")
        return self.dockq_data

    def create_batch(self, target_name, input_name):
        """
        根据 DockQ 分数和 CAPRI 类别分配正负样本，创建一个 batch
        :param target_name: 当前结构的目标名称
        :param input_name: 当前结构的输入名称
        :return: batch（包含正负样本的 decoys）
        """
        # 获取当前结构的 CAPRI 类别
        capri_category = self.dockq_data.loc[self.dockq_data['Model'] == input_name, 'CAPRI'].values[0]

        # 获取当前目标的所有 decoy
        target_data = self.dockq_data[self.dockq_data['Model'].str.startswith(target_name)]

        batch = [input_name]

        if capri_category == 'Incorrect':
            # 正样本从 'Incorrect' 类别选择
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Incorrect']['Model'].tolist())
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'Acceptable':
            # 正样本从 'Acceptable' 类别中选择
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Acceptable']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'Medium':
            # 正样本从 'Medium' 类别中选择
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'High']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'High':
            # 正样本从 'High' 类别中选择
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'High']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)

        return batch

    def select_negative_samples(self, target_name, capri_category):
        """
        选择负样本：
        1. **优先从相同 target 下选择负样本**（类别不同）
        2. **如果不足，再从其他 targets 下选择负样本**

        :param target_name: 当前 target 名称
        :param capri_category: 当前结构的 CAPRI 类别
        :return: 负样本 decoy 列表（最多 4 个）
        """
        negative_decoys = []

        # 1. **从相同 target 下选择类别不同的负样本**
        target_data = self.dockq_data[self.dockq_data['Model'].str.startswith(target_name)]
        candidate_negatives = target_data[target_data['CAPRI'] != capri_category]  # 选不同类别的

        num_needed = 2  # 目标负样本数量

        if len(candidate_negatives) > 0:
            selected_negatives = candidate_negatives.sample(n=min(num_needed, len(candidate_negatives)))
            negative_decoys.extend(selected_negatives['Model'].tolist())
            num_needed -= len(negative_decoys)

        # 2. **如果当前 target 内的负样本不足，再从其他 targets 选择**
        if num_needed > 0:
            other_targets = self.dockq_data[~self.dockq_data['Model'].str.startswith(target_name)]
            if capri_category == 'Incorrect':
                # Incorrect 的负样本应从 Acceptable, Medium, High 选
                extra_negatives = other_targets[other_targets['CAPRI'].isin(['Acceptable', 'Medium', 'High'])]
            else:
                # 其他类别的负样本应从 Incorrect 选
                extra_negatives = other_targets[other_targets['CAPRI'] == 'Incorrect']

            if len(extra_negatives) > 0:
                selected_extra_negatives = extra_negatives.sample(n=min(num_needed, len(extra_negatives)))
                negative_decoys.extend(selected_extra_negatives['Model'].tolist())

        return negative_decoys


class DGLDataset(Dataset):
    def __init__(self, data_folder, lddt_csv, file_list=None):
        """
        初始化 DGLDataset
        :param data_folder: 数据集所在文件夹路径
        :param lddt_csv: 包含 lDDT 分数数据的 CSV 文件路径
        :param file_list: 文件名列表（可选），如果指定则只加载该列表中的文件
        """
        self.data_folder = data_folder
        self.lddt_data = self.load_lddt_data(lddt_csv)

        if file_list:
            # 如果提供了文件名列表，只加载该列表中的文件
            self.dgl_files = [os.path.join(data_folder, f"{file_name}") for file_name in file_list]
        else:
            # 否则加载文件夹中的所有 .dgl 文件
            self.dgl_files = sorted(glob.glob(os.path.join(data_folder, '*.dgl')))

    def __len__(self):
        return len(self.dgl_files)

    def __getitem__(self, idx):
        """
        根据索引返回当前样本的图、分数、正负样本的 batch 和正负样本的 DGLGraph
        :param idx: 索引
        :return: 当前样本的图、评分、正样本、负样本及其图
        """
        # 获取当前文件的图和评分
        dgl_file = self.dgl_files[idx]
        dgl_graph, score, lddt_level, input_name, target_name = self.load_graph_and_score(dgl_file)

        # 返回当前样本的图、评分、正负样本的名称和图
        return dgl_graph, score, lddt_level

    def load_segmented_data(self, csv_path, score_type="lDDT"):
        """
        加载分段 CSV 文件
        :param csv_path: CSV 文件路径
        :param score_type: 分数类型 ("lDDT" 或其他)
        :return: 包含所有段落数据的 DataFrame
        """
        data_frames = []
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            current_data = []
            for line in lines:
                if line.startswith("==="):  # 新段落的标志
                    if current_data:  # 保存当前段落为 DataFrame
                        data_frames.append(pd.read_csv(StringIO("\n".join(current_data))))
                        current_data = []
                else:
                    current_data.append(line.strip())

            # 添加最后一段
            if current_data:
                data_frames.append(pd.read_csv(io.StringIO("\n".join(current_data))))

        # 合并所有段落
        combined_data = pd.concat(data_frames, ignore_index=True)
        if score_type not in combined_data.columns:
            raise ValueError(f"Expected column '{score_type}' not found in file: {csv_path}")
        return combined_data

    def load_graph_and_score(self, dgl_file):
        """
        加载 DGL 图和评分数据
        :param dgl_file: DGL 文件路径
        :return: 返回图结构、lDDT 评分、目标名称和 CAPRI 类别
        """
        # 加载 DGL 图
        dgl_graphs, _ = dgl.load_graphs(dgl_file)
        if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
            dgl_graph = dgl_graphs[0]
        else:
            raise ValueError(f"No DGL graph loaded from {dgl_file}")

        # 获取文件名并解析目标名称
        input_name = os.path.basename(dgl_file).split('.')[0]
        target_name = input_name.split('_decoy')[0]

        # 提取 decoy 索引
        decoy_index = int(re.search(r'(\d+)$', input_name).group(1))

        # 获取对应文件的评分
        score, lddt_level = self.get_lddt_score(target_name, decoy_index)

        return dgl_graph, score, lddt_level, input_name, target_name

    def get_lddt_score(self, target_name, decoy_index):
        """
        获取 lDDT 评分和 lDDT level
        :param target_name: 目标名称
        :param decoy_index: decoy 索引
        :return: 对应的 lDDT 评分 和 lDDT level
        """
        if self.lddt_data is None:
            raise ValueError("lDDT data has not been loaded.")

        model_name = f"{target_name}_decoy{decoy_index}"
        if 'Model' not in self.lddt_data.columns:
            raise KeyError("'Model' column is missing from lDDT data.")
        if 'lDDT level' not in self.lddt_data.columns:
            raise KeyError("'lDDT level' column is missing from lDDT data.")

        if model_name in self.lddt_data['Model'].values:
            row = self.lddt_data.loc[self.lddt_data['Model'] == model_name]
            score = row['lDDT'].values[0]
            level = row['lDDT level'].values[0]  # 直接读取 lDDT level

            return score, level
        else:
            raise ValueError(f"No lDDT score found for model {model_name}")

    def load_lddt_data(self, lddt_csv):
        """
        加载 lDDT 分数数据
        :param lddt_csv: 包含 lDDT 分数的 CSV 文件路径
        :return: lDDT 分数数据的 DataFrame
        """
        self.lddt_data = self.load_segmented_data(lddt_csv, score_type="lDDT")
        return self.lddt_data


class MLPReadoutlddtClass(nn.Module):
    """Read-out Module for Regression Task"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5):
        super(MLPReadoutlddtClass, self).__init__()
        self.list_FC_layer = nn.Sequential()

        # 第一层：从 input_dim 降到 256
        self.list_FC_layer.add_module(f'Linear_1', nn.Linear(input_dim, 256, bias=True))
        self.list_FC_layer.add_module(f'BN_1', nn.BatchNorm1d(256))
        self.list_FC_layer.add_module(f'relu_1', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_1', nn.Dropout(p=dp_rate))

        # 第二层：从 256 降到 128
        self.list_FC_layer.add_module(f'Linear_2', nn.Linear(256, 128, bias=True))
        self.list_FC_layer.add_module(f'BN_2', nn.BatchNorm1d(128))
        self.list_FC_layer.add_module(f'relu_2', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_2', nn.Dropout(p=dp_rate))

        # 第三层：从 128 降到 64
        self.list_FC_layer.add_module(f'Linear_3', nn.Linear(128, 64, bias=True))
        self.list_FC_layer.add_module(f'BN_3', nn.BatchNorm1d(64))
        self.list_FC_layer.add_module(f'relu_3', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_3', nn.Dropout(p=dp_rate))

        # 第四层：从 64 降到 1
        self.last_layer_classification = nn.Linear(64, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_class = self.last_layer_classification(x)  # class label
        # 输出单个预测值
        # y_pred = x.squeeze(-1)  # 去掉多余的维度 (1,)

        # 输出范围在 [0, 1] 之间
        y_pred = torch.sigmoid(self.last_layer(y_class))

        return y_pred, y_class


class MLPReadoutlddtClassV2(nn.Module):
    """Read-out Module for Regression Task"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5):
        super().__init__()

        # Attention 模块
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(input_dim)

        # 原来的 MLP 部分
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=dp_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dp_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=dp_rate),
        )

        self.class_layer = nn.Linear(64, 4)
        self.output_layer = nn.Linear(4, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim) → reshape for attention
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        x = self.attn_norm(attn_out + x)  # Residual + Norm
        x = x.squeeze(1)  # (batch_size, input_dim)

        # MLP
        x = self.mlp(x)
        y_class = self.class_layer(x)
        y_pred = torch.sigmoid(self.output_layer(y_class))

        return y_pred, y_class


class MLPReadoutlddtClassV3(nn.Module):
    """Read-out Module for Regression Task"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5):
        super(MLPReadoutlddtClassV3, self).__init__()
        self.list_FC_layer = nn.Sequential()

        # 第一层：从 input_dim 降到 256
        self.list_FC_layer.add_module(f'Linear_1', nn.Linear(input_dim, 256, bias=True))
        self.list_FC_layer.add_module(f'BN_1', nn.BatchNorm1d(256))
        self.list_FC_layer.add_module(f'relu_1', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_1', nn.Dropout(p=dp_rate))

        # 第二层：从 256 降到 128
        self.list_FC_layer.add_module(f'Linear_2', nn.Linear(256, 128, bias=True))
        self.list_FC_layer.add_module(f'BN_2', nn.BatchNorm1d(128))
        self.list_FC_layer.add_module(f'relu_2', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_2', nn.Dropout(p=dp_rate))

        # 第三层：从 128 降到 64
        self.list_FC_layer.add_module(f'Linear_3', nn.Linear(128, 64, bias=True))
        self.list_FC_layer.add_module(f'BN_3', nn.BatchNorm1d(64))
        self.list_FC_layer.add_module(f'relu_3', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_3', nn.Dropout(p=dp_rate))

        # 第三层：从 64 降到 32
        self.list_FC_layer.add_module(f'Linear_4', nn.Linear(64, 32, bias=True))
        self.list_FC_layer.add_module(f'BN_4', nn.BatchNorm1d(32))
        self.list_FC_layer.add_module(f'relu_4', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_4', nn.Dropout(p=dp_rate))

        # 第四层：从 32 降到 1
        self.last_layer_classification = nn.Linear(32, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_class = self.last_layer_classification(x)  # class label
        # 输出单个预测值
        # y_pred = x.squeeze(-1)  # 去掉多余的维度 (1,)

        # 输出范围在 [0, 1] 之间
        y_pred = torch.sigmoid(self.last_layer(y_class))

        return y_pred, y_class


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

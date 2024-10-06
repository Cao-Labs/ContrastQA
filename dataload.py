import os

import dgl
import keras
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import time
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder


# 返回预先计算并保存的特征排名
from torch import nn


def get_feature_ranks(pathToFeatureScores):
    '''
    This method returns the ordered feature ranks, precomputed and saved

    Parameters:
    -------------
    pathToFeatureScores: string
        This is a hard-coded path, points to the Pearson_Correlation_Features.txt

    Returns:
    -----------
    list: [string]
        A list of strings of the feature indexes ranked from best to worst by pearson correlation
    '''
    # load and parse the feature ranks

    raw_data = open(pathToFeatureScores).read()
    feature_ranks = []
    for line in raw_data.split('\n'):
        line_data = line.split('\t')
        feature_ranks.append(line_data[0])

    return feature_ranks


# 定义MLP线性降维类
class ESMEmbeddingDimReducer(nn.Module):
    def __init__(self, input_dim=1280, output_dim=300):
        super(ESMEmbeddingDimReducer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# 加载ESM嵌入特征并进行降维
def load_esm_embeddings(esm_path, pdb_name, reducer):
    """
    加载ESM嵌入特征，并通过MLP线性层将特征维度从1280降到300。

    Parameters:
    -----------
    esm_path : str
        ESM嵌入特征的目录路径。
    pdb_name : str
        PDB文件名，用于定位相关ESM嵌入特征。
    reducer : nn.Module
        用于降维的MLP线性层。

    Returns:
    --------
    dict: ESM嵌入特征，按链分组并降维。
    """
    esm_file_path = os.path.join(esm_path, f"{pdb_name}_emb.pkl")
    with open(esm_file_path, 'rb') as f:
        esm_data = pickle.load(f)

    reduced_esm_data = {}

    # 遍历每条链，将嵌入特征降维
    for chain, layers in esm_data.items():
        reduced_esm_data[chain] = {}
        for layer, embedding in layers.items():
            embedding_tensor = torch.tensor(embedding)  # 将numpy数组转换为tensor
            reduced_embedding = reducer(embedding_tensor)  # 通过MLP降维
            reduced_esm_data[chain][layer] = reduced_embedding.detach().numpy()  # 转回numpy

    return reduced_esm_data


class ProteinGVPQAData_Generator(tf.keras.utils.Sequence):
    def __init__(self, data_path='../data/temp_out/DG/pkl_output', esm_path='../data/temp_out/DG_embedding_pkl',
                 protein_info_pickle='../data/temp_out/MUL_info/MUL_1_info.csv', min_seq_size=0,
                 max_seq_size=10000, batch_size=1, max_msa_seq=100000, max_id_nums=1000000):
        self.data_path = data_path
        self.protein_info = protein_info_pickle
        self.max_id_nums = max_id_nums
        self.min_seq_size = min_seq_size
        self.max_seq_size = max_seq_size
        self.batch_size = batch_size
        self.max_msa_seq = max_msa_seq
        self.contact_json_path = self.data_path
        self.id2seq, self.id2interfacemask = self.get_filenames(head=self.max_id_nums)
        self.id_list = list(self.id2seq.keys())

        self.esm_path = esm_path
        self.reducer = ESMEmbeddingDimReducer(input_dim=1280, output_dim=300)

        # 直接将batchsize设置为1
        batchsize = 1
        self.seq2id = defaultdict(list)
        for seq_id, seq_len in sorted(self.id2seq.items()):
            self.seq2id[seq_len].append(seq_id)

        protein_model_id_list = []

        for seq_len in self.seq2id.keys():
            comb = self.seq2id[seq_len]
            batch_len = int(len(comb) / batchsize) + 1
            for index in range(batch_len):
                batch_list = comb[index * batchsize: (index + 1) * batchsize]
                protein_model_id_list.append(batch_list)

        self.protein_model_id_list = protein_model_id_list
        self.on_epoch_begin()

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.protein_model_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.protein_model_id_list) / self.batch_size))

    def __getitem__(self, index):
        batch_id_list = self.protein_model_id_list[index]
        if len(batch_id_list) == 0:
            batch_id_list = random.sample(self.protein_model_id_list, 1)[0]

        node_feat_batch, model_contact_batch, pdb_distance_pair_batch = self.collect_data(batch_id_list)

        return node_feat_batch, model_contact_batch, pdb_distance_pair_batch

    def __call__(self):
        self.i = 0
        return self

    def get_filenames(self, head=10000000):
        model2seq = {}
        model2interfacemask = {}

        start = time.time()
        train_dataframe = pd.read_csv(self.protein_info, sep=',')

        protein_list = set()
        for i in train_dataframe.index:
            pdb_id = train_dataframe['target'][i]
            model_id = train_dataframe['model'][i]
            model_interfacemask = eval(train_dataframe['interface_mask'][i])
            seq_len = len(model_interfacemask)

            protein_list.add(pdb_id)

            if seq_len <= self.max_seq_size and seq_len >= self.min_seq_size:
                model2seq[model_id] = seq_len
                model2interfacemask[model_id] = model_interfacemask

            if len(model2seq) >= head:
                break

        end = time.time()
        print("Finish loading ", self.protein_info, " takes ", end - start)

        return model2seq, model2interfacemask

    def get_node_edges_features(self, pdb_name, esm_path, reducer):
        """
        加载普通节点特征和ESM嵌入特征，并进行特征相加处理。

        Parameters:
        -----------
        pdb_name : str
            PDB文件名，用于定位普通节点特征。
        esm_path : str
            ESM嵌入特征文件目录路径。
        reducer : nn.Module
            用于降维的MLP线性层（如果需要）。

        Returns:
        --------
        tuple: 返回相加后的节点特征、距离矩阵和接口掩码。
        """
        target_id = pdb_name.split('_')[0]  # 提取PDB ID
        pdb_index = pdb_name.split('.')[0]  # 去掉文件后缀
        node_features = []
        esm_features = []

        # 普通特征文件路径
        feature_path = os.path.join(self.data_path, f'{pdb_index}.pkl')

        # ESM嵌入特征文件路径
        esm_pdb_index = pdb_index + '_emb'  # 对应 ESM 嵌入特征文件的命名格式
        esm_file_path = os.path.join(esm_path, f'{esm_pdb_index}.pkl')

        # 加载普通节点特征
        # feature_ranks = get_feature_ranks('Pearson_Correlation_Features.txt')  # training set
        feature_ranks = get_feature_ranks('./data/Pearson_Correlation_Features.txt')  # test set

        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
            distance_map = []
            for index in data:
                # 提取普通特征
                aa_density_change = data[index]['aa_density_change']
                aa_density = aa_density_change.flatten('F')
                avg_distance = data.get('average_distance', np.zeros(aa_density.shape))  # 如果没有distance则设置默认值
                hydro_change = data[index]['hydro_change']
                iso_change = data[index]['iso_change']
                mass_change = data[index]['mass_change']
                per_contact = data[index]['percent_contact']
                sol_change = data[index]['sol_change']
                std_dev_distance = data[index]['std_dev_distance']
                structure_contact_matrix = data[index]['structure_contact_matrix']
                structure_contact = structure_contact_matrix.flatten('F')

                X = np.concatenate([aa_density, avg_distance, hydro_change, iso_change, mass_change, per_contact,
                                    sol_change, std_dev_distance, structure_contact]).astype(np.float32)

                filter_feature = [X[int(feature_number)] for feature_number in feature_ranks[:300]]
                node_features.append(filter_feature)
                distance_map.append(data[index]['contact_map'])

        # 加载ESM嵌入特征
        if os.path.exists(esm_file_path):  # 确保 ESM 文件存在
            esm_embeddings = load_esm_embeddings(esm_path, pdb_index, reducer)
            for chain, layers in esm_embeddings.items():
                for layer, embedding in layers.items():
                    # 确保 ESM 嵌入特征的形状一致
                    esm_features.append(np.array(embedding))
        else:
            print(f"Warning: ESM embedding file not found for {pdb_name}")
            return None

        # 将 ESM 嵌入特征降维到与普通节点特征相同的维度（如果需要）
        if esm_features:
            esm_features = np.vstack(esm_features)  # 将 ESM 特征合并为一个大数组
            if esm_features.shape[0] == len(node_features):
                if esm_features.shape[1] != len(node_features[0]):
                    print(
                        "Warning: ESM features dimension does not match node features dimension. Applying dimensionality reduction.")
                    esm_features = reducer(torch.tensor(esm_features, dtype=torch.float32)).detach().numpy()

                # 确保特征维度匹配后进行相加
                node_features = np.array(node_features)
                if node_features.shape[1] != esm_features.shape[1]:
                    print(
                        f"Warning: Node features dimension {node_features.shape[1]} does not match ESM features dimension {esm_features.shape[1]}.")
                node_features += esm_features
            else:
                print("Warning: ESM features and node features do not match in number of nodes.")
        else:
            node_features = np.array(node_features)

        # 将结果转换为 NumPy 数组
        node_features_array = np.array(node_features)
        distance_map_array = np.array(distance_map)

        # 获取接口掩码
        local_interfacemask = np.array(self.id2interfacemask[pdb_index + '.pdb'.strip()])
        if len(local_interfacemask) != node_features_array.shape[0]:
            print(
                f"Warning: Interface mask length {len(local_interfacemask)} does not match node features length {node_features_array.shape[0]}.")

        return node_features_array, distance_map_array, local_interfacemask

    def collect_data(self, batch_id_list):
        max_seq_len = 0
        for i in range(len(batch_id_list)):
            try:
                if self.id2seq[batch_id_list[i]] > max_seq_len:
                    max_seq_len = self.id2seq[batch_id_list[i]]
            except:
                print("error batch_id_list[i]: ", batch_id_list[i])
                exit(-1)

        self.pad_size = max_seq_len

        pdb_node_batch = np.full((len(batch_id_list), self.pad_size, 300), 0.0)
        model_contact_batch = np.full((len(batch_id_list), self.pad_size, self.pad_size, 1), 0.0)
        pdb_distance_pair_batch = np.full((len(batch_id_list), self.pad_size, self.pad_size, 1), 0.0)
        model_aa_interfacemask_batch = np.full((len(batch_id_list), self.pad_size, 1), 0.0)

        updated_max_seq_len = 0
        target2models = defaultdict(list)
        target2models_seq = defaultdict(list)
        for i in range(len(batch_id_list)):
            try:
                info = batch_id_list[i].split('.')
            except:
                print("batch_id_list[i]: ", batch_id_list[i])
                exit(-1)
            seq_len = self.id2seq[batch_id_list[i]]
            target_id = info[0].split('_')[0]

            target2models[target_id].append(batch_id_list[i])
            target2models_seq[target_id].append(seq_len)

        loaded_count = 0
        for target_id in target2models.keys():
            models = target2models[target_id]

            seq_len_unique = np.unique(target2models_seq[target_id])

            if len(seq_len_unique) > 1:
                print('Warning: pdb structures from same target have different length, check it')
                exit(-1)
            seq_len = seq_len_unique[0]

            for model in models:
                node_features, pdb_distance_feature, node_interface = self.get_node_edges_features(model, self.esm_path,
                                                                                                   self.reducer)

                l = len(pdb_distance_feature)

                if l != seq_len:
                    print("warning: pdb length is not equal to sequence length", l, "!=", seq_len)

                if l != len(node_interface):
                    print("warning: pdb length is not equal to sequence interface length", l, "!=", node_interface)

                contact_feature = pdb_distance_feature.copy()
                contact_feature[contact_feature < 8] = 1
                contact_feature[contact_feature >= 8] = 0
                contact_feature = contact_feature.astype(np.uint8)

                np.fill_diagonal(contact_feature, 1)

                model_distance = np.multiply(contact_feature, pdb_distance_feature)

                interface = set(np.where(node_interface == 1)[0].flatten())
                for row in range(len(interface)):
                    expand_set = set(np.where(pdb_distance_feature[node_interface != 0, :][row] < 0)[0].flatten())
                    interface = interface.union(expand_set)

                node_interface[list(interface)] = 1
                interface_mask = np.array(node_interface).astype(float)

                interface_size = np.sum(interface_mask == 1)

                filter_node_features = node_features[interface_mask != 0, :]

                filter_contact_feature = contact_feature[interface_mask != 0, interface_mask != 0]
                filter_model_distance = model_distance[interface_mask != 0, interface_mask != 0]

                if updated_max_seq_len < interface_size:
                    updated_max_seq_len = interface_size

                pdb_node_batch[loaded_count, 0:interface_size, :] = filter_node_features
                model_contact_batch[loaded_count, 0:interface_size, 0:interface_size, 0] = filter_contact_feature
                pdb_distance_pair_batch[loaded_count, 0:interface_size, 0:interface_size, 0] = filter_model_distance
                loaded_count += 1

        return pdb_node_batch, model_contact_batch, pdb_distance_pair_batch

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            lambda: self,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 300), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32)
            )
        )

    def convert_to_dgl(self, node_features, contact_map):
        graphs = []
        for i in range(len(node_features)):
            num_nodes = node_features[i].shape[0]

            try:
                contact_map_reshaped = contact_map[i].reshape(num_nodes, num_nodes)
            except ValueError as e:
                print(f"Error reshaping contact_map[{i}]: {e}")
                continue

            # 获取非零元素的索引
            u, v = np.nonzero(contact_map_reshaped)

            # 确保有边存在
            if len(u) == 0 or len(v) == 0:
                print(f"No edges found for graph {i}.")
                continue

            # 输出调试信息
            print(f"node_features[{i}] shape: {node_features[i].shape}")
            print(f"contact_map[{i}] shape: {contact_map[i].shape}")

            g = dgl.graph((u, v), num_nodes=num_nodes)

            # 标准化节点特征
            node_feats = torch.tensor(node_features[i], dtype=torch.float32)
            min_val = node_feats.min()
            max_val = node_feats.max()
            min_val = max(min_val, 0.0001)
            max_val = max(max_val, 0.0001)
            node_feats = (node_feats - min_val) / (max_val - min_val)

            # 只在存在边的位置选择边特征
            edge_feats = torch.tensor(contact_map_reshaped[u, v], dtype=torch.float32).unsqueeze(1)
            g.ndata['feat'] = node_feats
            g.edata['weight'] = edge_feats

            graphs.append(g)

        return graphs

    def save_dgl_graphs(self, dgl_graphs, filename):
        dgl.save_graphs(filename, dgl_graphs)
        print(f"Graphs saved to {filename}")

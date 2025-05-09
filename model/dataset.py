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

        self.data_folder = data_folder
        self.lddt_data = self.load_lddt_data(lddt_csv)
        self.dockq_data = self.load_dockq_data(dockq_csv)

        if file_list:
            self.dgl_files = [os.path.join(data_folder, f"{file_name}") for file_name in file_list]
        else:
            self.dgl_files = sorted(glob.glob(os.path.join(data_folder, '*.dgl')))

        self.batch_size = batch_size

    def __len__(self):
        return len(self.dgl_files)  

    def __getitem__(self, idx):

        # return adl graph and label
        dgl_file = self.dgl_files[idx]
        dgl_graph, lddt_score, lddt_level, input_name, target_name, \
        dockq_score, capri_category = self.load_graph_and_score(dgl_file)

        # create batch including positive and negative samples' names
        cluster_batch = self.create_batch(target_name, input_name)

        # divide positive and negative samples（cluster_batch[0] is anchor sample）
        positive_sample = cluster_batch[1]  # positive sample
        negative_samples = cluster_batch[2:]  # negative samples

        # load positive graph
        positive_graph_path = os.path.join(self.data_folder, f"{positive_sample}.dgl")
        positive_graph, _ = dgl.load_graphs(positive_graph_path)
        positive_graph = positive_graph[0]

        # positive samples DockQ score & target_name
        pos_dockq_score = self.dockq_data.loc[self.dockq_data['Model'] == positive_sample, 'DockQ'].values[0]
        pos_target = positive_sample.split('_decoy')[0]

        # negative samples
        negative_graphs = []
        neg_dockq_scores = []
        neg_targets = []

        for model_name in negative_samples:
            graph_path = os.path.join(self.data_folder, f"{model_name}.dgl")
            graphs, _ = dgl.load_graphs(graph_path)
            negative_graphs.append(graphs[0])

            # negative samples DockQ score & target_name
            neg_score = self.dockq_data.loc[self.dockq_data['Model'] == model_name, 'DockQ'].values[0]
            neg_dockq_scores.append(neg_score)

            neg_target = model_name.split('_decoy')[0]
            neg_targets.append(neg_target)

        return (dgl_graph, lddt_score, lddt_level, dockq_score, target_name,
                positive_graph, pos_dockq_score, pos_target,
                negative_graphs, neg_dockq_scores, neg_targets)

    def load_segmented_data(self, csv_path, score_type="lDDT"):

        data_frames = []
        with open(csv_path, 'r') as file:
            lines = file.readlines()
            current_data = []
            for line in lines:
                if line.startswith("==="):  
                    if current_data:  
                        data_frames.append(pd.read_csv(StringIO("\n".join(current_data))))
                        current_data = []
                else:
                    current_data.append(line.strip())

            if current_data:
                data_frames.append(pd.read_csv(io.StringIO("\n".join(current_data))))

        combined_data = pd.concat(data_frames, ignore_index=True)
        if score_type not in combined_data.columns:
            raise ValueError(f"Expected column '{score_type}' not found in file: {csv_path}")
        return combined_data

    def load_graph_and_score(self, dgl_file):

        # load DGL graph
        dgl_graphs, _ = dgl.load_graphs(dgl_file)
        if isinstance(dgl_graphs, list) and len(dgl_graphs) > 0:
            dgl_graph = dgl_graphs[0]
        else:
            raise ValueError(f"No DGL graph loaded from {dgl_file}")

        # get file name
        input_name = os.path.basename(dgl_file).split('.')[0]
        target_name = input_name.split('_decoy')[0]

        # extract decoy index
        decoy_index = int(re.search(r'(\d+)$', input_name).group(1))

        # get lddt score and level
        score, lddt_level = self.get_lddt_score(target_name, decoy_index)

        # get DockQ score and CAPRI category
        dockq_row = self.dockq_data.loc[self.dockq_data['Model'] == input_name]
        if dockq_row.empty:
            raise ValueError(f"No DockQ data found for model {input_name}")
        dockq_score = dockq_row['DockQ'].values[0]
        capri_category = dockq_row['CAPRI'].values[0]

        return dgl_graph, score, lddt_level, input_name, target_name, dockq_score, capri_category

    def get_lddt_score(self, target_name, decoy_index):

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
            level = row['lDDT level'].values[0]  # lDDT level

            return score, level
        else:
            raise ValueError(f"No lDDT score found for model {model_name}")

    def load_lddt_data(self, lddt_csv):

        self.lddt_data = self.load_segmented_data(lddt_csv, score_type="lDDT")
        return self.lddt_data

    def load_dockq_data(self, dockq_csv):

        self.dockq_data = self.load_segmented_data(dockq_csv, score_type="DockQ")
        return self.dockq_data

    def create_batch(self, target_name, input_name):

        # get CAPRI category
        capri_category = self.dockq_data.loc[self.dockq_data['Model'] == input_name, 'CAPRI'].values[0]

        # get all decoys of the target
        target_data = self.dockq_data[self.dockq_data['Model'].str.startswith(target_name)]

        batch = [input_name]

        if capri_category == 'Incorrect':
            # choose positive sample from 'Incorrect' category
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Incorrect']['Model'].tolist())
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'Acceptable':
            # choose positive sample from 'Acceptable' category
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Acceptable']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'Medium':
            # choose positive sample from 'Medium' category
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'High']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)
        elif capri_category == 'High':
            # choose positive sample from 'High' category
            positive_decoy = random.choice(target_data[target_data['CAPRI'] == 'High']['Model'].tolist())
            if not positive_decoy:
                positive_decoy = target_data[target_data['CAPRI'] == 'Medium']['Model'].tolist()
            negative_decoys = self.select_negative_samples(target_name, capri_category)
            batch.extend([positive_decoy] + negative_decoys)

        return batch

    def select_negative_samples(self, target_name, capri_category):

        negative_decoys = []

        # 1. **choose negative samples of different categories under the same target **
        target_data = self.dockq_data[self.dockq_data['Model'].str.startswith(target_name)]
        candidate_negatives = target_data[target_data['CAPRI'] != capri_category]  

        num_needed = 2  # number of negative samples

        if len(candidate_negatives) > 0:
            selected_negatives = candidate_negatives.sample(n=min(num_needed, len(candidate_negatives)))
            negative_decoys.extend(selected_negatives['Model'].tolist())
            num_needed -= len(negative_decoys)

        # 2. **if the number of negative samples is deficient under the same target，choose negatives from different targets**
        if num_needed > 0:
            other_targets = self.dockq_data[~self.dockq_data['Model'].str.startswith(target_name)]
            if capri_category == 'Incorrect':
                # the negative samples of Incorrect should choose from 'Acceptable, Medium, High' category
                extra_negatives = other_targets[other_targets['CAPRI'].isin(['Acceptable', 'Medium', 'High'])]
            else:
                extra_negatives = other_targets[other_targets['CAPRI'] == 'Incorrect']

            if len(extra_negatives) > 0:
                selected_extra_negatives = extra_negatives.sample(n=min(num_needed, len(extra_negatives)))
                negative_decoys.extend(selected_extra_negatives['Model'].tolist())

        return negative_decoys


class MLPReadoutlddtClass(nn.Module):
    """Read-out Module for Regression Task"""

    def __init__(self, input_dim: int, output_dim: int, dp_rate=0.5):
        super(MLPReadoutlddtClass, self).__init__()
        self.list_FC_layer = nn.Sequential()

        # first：from input_dim to 256
        self.list_FC_layer.add_module(f'Linear_1', nn.Linear(input_dim, 256, bias=True))
        self.list_FC_layer.add_module(f'BN_1', nn.BatchNorm1d(256))
        self.list_FC_layer.add_module(f'relu_1', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_1', nn.Dropout(p=dp_rate))

        # second：from 256 to 128
        self.list_FC_layer.add_module(f'Linear_2', nn.Linear(256, 128, bias=True))
        self.list_FC_layer.add_module(f'BN_2', nn.BatchNorm1d(128))
        self.list_FC_layer.add_module(f'relu_2', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_2', nn.Dropout(p=dp_rate))

        # third：from 128 to 64
        self.list_FC_layer.add_module(f'Linear_3', nn.Linear(128, 64, bias=True))
        self.list_FC_layer.add_module(f'BN_3', nn.BatchNorm1d(64))
        self.list_FC_layer.add_module(f'relu_3', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_3', nn.Dropout(p=dp_rate))

        # fourth：from 64 to 32
        self.list_FC_layer.add_module(f'Linear_4', nn.Linear(64, 32, bias=True))
        self.list_FC_layer.add_module(f'BN_4', nn.BatchNorm1d(32))
        self.list_FC_layer.add_module(f'relu_4', nn.LeakyReLU())
        self.list_FC_layer.add_module(f'dp_4', nn.Dropout(p=dp_rate))

        self.last_layer_classification = nn.Linear(32, 4, bias=True)
        self.last_layer = nn.Linear(4, output_dim, bias=True)

    def forward(self, x):
        x = self.list_FC_layer(x)
        y_class = self.last_layer_classification(x)  # class label
        y_pred = torch.sigmoid(self.last_layer(y_class)) # score

        return y_pred, y_class

"""
@ Description: helper functions
"""

import os
from copy import deepcopy
import pandas as pd
import torch
from Bio import PDB
import dgl
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from .general import exists
from .pdb import get_atom_coords
import numpy as np
from biopandas.pdb import PandasPdb
from typing import List, Union
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import Dataset
import torch.nn.functional as F


def laplacian_positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> torch.Tensor:
    """
        Graph positional encoding v/ Laplacian eigenvectors
        :return torch.Tensor (L, pos_enc_dim)
    """

    # Laplacian
    A = g.adjacency_matrix()
    s = torch.sparse_coo_tensor(indices=A.coalesce().indices(),
                                values=A.coalesce().val,
                                size=A.coalesce().shape)
    A = s.to_dense()
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.A)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    laplacian_feature = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().reshape(-1, pos_enc_dim)
    return laplacian_feature


def ss3_one_hot(df: pd.DataFrame) -> torch.Tensor:
    """treat ss8 to ss3 get one hot encoding, return size L * 3"""
    tokens_dict = {'H': 0, 'B': 2, 'E': 2, 'G': 0, 'I': 0, 'T': 1, 'S': 1, '-': 1}
    one_hot_array = np.zeros([df.shape[0], 3])
    ss8_list = df.ss8.to_list()

    for idx, item in enumerate(ss8_list):
        if item not in tokens_dict:
            raise KeyError(f'This {item} is not secondary structure type.')
        col_idx = tokens_dict[item]
        one_hot_array[idx, col_idx] = 1

    return torch.from_numpy(one_hot_array).reshape(-1, 3)


def pdb2fasta(pdb_file: str) -> str:
    """extract sequence from pdb file"""

    amino = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
             'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
             'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
             'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}

    if not os.path.isfile(pdb_file):
        raise FileExistsError(f'PDB File does not exist {pdb_file}')

    with open(pdb_file, 'r') as file:
        content = file.readlines()

    seq = []
    prev_mark = -1

    for line in content:
        if line[:4] == 'ATOM':
            pos_mark = line[22: 26].strip()
            if pos_mark != prev_mark:
                seq.append(amino[line[17:20]])
            prev_mark = pos_mark

    return "".join(seq)


def read_fasta(fasta_file: str):
    """
    Read fasta file, return sequence id, length and content
    Support Fasta format example:
    >Target_id|length
    CCCCCCCCCCCCCCCCC
    """
    with open(fasta_file) as f:
        content = f.readlines()
        f.close()
    seq = ''
    if len(content) == 1:
        seq += content[0].strip()
        return seq
    else:
        target_id, length = content[0].split('|')
        target_id = target_id.strip().strip('>')
        length = int(length.strip())
        seq += content[1].strip()
    return target_id, length, seq


def sequence_one_hot(fasta_file: str) -> torch.Tensor:
    """Sequence one hot encoding, return size L * 21"""
    tokens_dict_regular_order = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
                                 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                                 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
                                 'R': 14, 'S': 15, 'T': 16, 'V': 17,
                                 'W': 18, 'Y': 19, 'X': 20}

    seq = fasta_file
    length = len(seq)

    one_hot_array = np.zeros([length, 21])

    for idx, item in enumerate(seq.upper()):
        if item not in tokens_dict_regular_order.keys():
            item = 'X'
        col_idx = tokens_dict_regular_order[item]
        one_hot_array[idx, col_idx] = 1

    return torch.from_numpy(one_hot_array).reshape(-1, 21)


def edge_sin_pos(g: dgl.DGLGraph) -> torch.Tensor:
    """Edge wise encoding"""
    return torch.sin((g.edges()[0] - g.edges()[1]).float()).reshape(-1, 1)


def update_node_feature(graph: dgl.DGLGraph, new_node_features: List) -> None:
    """Node feature update helper"""
    for node_feature in new_node_features:
        # print(f"Expected number of nodes: {graph.num_nodes()}, feature size: {node_feature.shape[0]}")
        if not graph.ndata:
            assert node_feature.shape[0] == graph.num_nodes()
            graph.ndata['feat'] = node_feature
        else:
            assert node_feature.shape[0] == graph.num_nodes()
            graph.ndata['feat'] = torch.cat((graph.ndata['feat'], node_feature), dim=1)


def update_edge_feature(graph: dgl.DGLGraph, new_edge_features: List) -> None:
    """Edge feature update helper"""
    for edge_feature in new_edge_features:
        if not graph.edata:
            graph.edata['feat'] = edge_feature
        else:
            graph.edata['feat'] = torch.cat((graph.edata['feat'], edge_feature), dim=1)
    return None


def remove_n(lst: List, pattern='\n') -> List:
    return [i.strip(pattern) for i in lst]


def txt_to_list(txt_file: str, pattern='\n') -> List:
    """read txt file to list"""
    with open(txt_file, 'r') as f:
        tmp_list = f.readlines()
    tmp_list = remove_n(tmp_list, pattern=pattern)
    return tmp_list


def list_to_txt(lst: List, txt_file: str) -> None:
    """read out list to txt file"""
    with open(txt_file, 'w') as f:
        for i in lst:
            f.writelines(i + '\n')
    return None


def pdb2graph_new_chain_info(pdb_file: str, knn=10):
    """
    Build KNN graph for a protein, return graph and src vertex
    and end dst vertex for graph, without self-loop, PEEE chain ID
    """
    atom_df = PandasPdb().read_pdb(pdb_file).df['ATOM']
    atom_df_full = deepcopy(atom_df)  # return all atom df for distance calculation
    atom_df = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    node_coords = torch.tensor(atom_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)
    protein_graph = dgl.knn_graph(node_coords, knn)
    protein_graph = protein_graph.remove_self_loop()  # remove self loop

    srcs = protein_graph.edges()[0]
    dsts = protein_graph.edges()[1]

    edges = list(zip(srcs, dsts))

    # CA-CA distance
    atom_df_ca = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    chain_id_list = atom_df_ca.loc[:, 'chain_id'].tolist()
    chain_id_dict = dict(zip([i for i in range(len(chain_id_list))], chain_id_list))  # {idx: chain_id}

    uniform_chain_feature = []
    for i in edges:
        u, v = i
        u = u.item()
        v = v.item()
        if (chain_id_dict[u] == chain_id_list[v]) and (abs(u - v) == 1):
            uniform_chain_feature.append(1)
        else:
            uniform_chain_feature.append(0)
    return atom_df_full, protein_graph, edges, torch.tensor(uniform_chain_feature).reshape(-1, 1)


def distance_helper(pdb_file: str, pdb_name: str,
                    output_folder: str, atom_type='CB',
                    save_flag=True) -> Union[tuple, np.ndarray]:
    """Calculate CA-CA, or CB-CB or N-O distance for a pdb file"""
    ppdb = PandasPdb().read_pdb(pdb_file)
    test_df = ppdb.df['ATOM']

    if atom_type == 'CB':
        # GLY does not have CB, use CA to instead of.
        filtered_df = test_df[((test_df.loc[:, 'residue_name'] == 'GLY') & (test_df.loc[:, 'atom_name'] == 'CA')) \
                              | (test_df.loc[:, 'atom_name'] == 'CB')]
    elif atom_type == 'CA':
        filtered_df = test_df[test_df.loc[:, 'atom_name'] == 'CA']
    elif atom_type == 'NO':
        filtered_df = test_df[(test_df.loc[:, 'atom_name'] == 'N') | (test_df.loc[:, 'atom_name'] == 'O')]
    else:
        raise ValueError('Atom type should be CA, CB or NO.')

    if atom_type != 'NO':
        coord = filtered_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        real_dist = euclidean_distances(coord)
    else:
        coord_N = filtered_df[filtered_df.loc[:, 'atom_name'] == 'N'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        coord_O = filtered_df[filtered_df.loc[:, 'atom_name'] == 'O'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        real_dist = euclidean_distances(coord_N, coord_O)  # up-triangle N-O, low-triangle O-N

    real_dist = np.round(real_dist, 3)

    if save_flag:
        np.save(file=os.path.join(output_folder, pdb_name + f'_{atom_type}.npy'), arr=real_dist)
        return real_dist.shape
    else:
        return real_dist

# extraction of tip atom

AA_to_tip = {"ALA": "CB", "CYS": "SG", "ASP": "CG", "ASN": "CG", "GLU": "CD",
             "GLN": "CD", "PHE": "CZ", "HIS": "NE2", "ILE": "CD1", "GLY": "CA",
             "LEU": "CG", "MET": "SD", "ARG": "CZ", "LYS": "NZ", "PRO": "CG",
             "VAL": "CB", "TYR": "OH", "TRP": "CH2", "SER": "OG", "THR": "OG1"}


def get_distmaps(pdb_file, atom1="CA", atom2="CA", default="CA"):
    parser = PDB.PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # 获取所有原子的坐标
    atom_coords1 = []
    atom_coords2 = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # 处理 atom1
                if isinstance(atom1, str):
                    if residue.has_id(atom1):
                        atom_coords1.append(residue[atom1].get_coord())
                    else:
                        atom_coords1.append(residue[default].get_coord())
                else:
                    atom_name = atom1.get(residue.get_resname(), default)
                    if residue.has_id(atom_name):
                        atom_coords1.append(residue[atom_name].get_coord())
                    else:
                        atom_coords1.append(residue[default].get_coord())

                # 处理 atom2
                if isinstance(atom2, str):
                    if residue.has_id(atom2):
                        atom_coords2.append(residue[atom2].get_coord())
                    else:
                        atom_coords2.append(residue[default].get_coord())
                else:
                    atom_name = atom2.get(residue.get_resname(), default)
                    if residue.has_id(atom_name):
                        atom_coords2.append(residue[atom_name].get_coord())
                    else:
                        atom_coords2.append(residue[default].get_coord())

    # 转换为 NumPy 数组
    xyz1 = np.array(atom_coords1)
    xyz2 = np.array(atom_coords2)

    # 计算距离矩阵
    return cdist(xyz1, xyz2)


def process_model(
    pdb_file,
    fasta_file=None,
    ignore_cdrs=None,
    ignore_chain=None,
):
    temp_coords, temp_mask = None, None
    if exists(pdb_file):
        temp_coords = get_atom_coords(
            pdb_file,
            fasta_file=fasta_file,
        )
        temp_coords = torch.stack(
            [
                temp_coords['N'], temp_coords['CA'], temp_coords['C'],
                temp_coords['CB']
            ],
            dim=1,
        ).view(-1, 3).unsqueeze(0)

        temp_mask = torch.ones(temp_coords.shape[:2]).bool()
        temp_mask[temp_coords.isnan().any(-1)] = False
        temp_mask[temp_coords.sum(-1) == 0] = False

    return temp_coords, temp_mask


def edge_positional_embeddings(self, edge_index,
                           num_embeddings=None,
                           period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or self.num_positional_embeddings
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))



import dgl
import numpy as np
import torch
from Bio import PDB
from scipy.spatial.distance import cdist
from .general import exists
import scipy.sparse as sp
from .pdb import get_atom_coords
import torch.nn.functional as F

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


def laplacian_positional_encoding(g: dgl.DGLGraph, pos_enc_dim: int) -> torch.Tensor:
    """
    Graph positional encoding via Laplacian eigenvectors
    :return torch.Tensor (L, pos_enc_dim)
    """

    # 获取邻接矩阵的稀疏表示
    A = g.adjacency_matrix()  # 默认获取稀疏矩阵
    A = A.coalesce()  # 获取稀疏矩阵的非零元素

    # 获取稀疏矩阵的索引和值
    indices = A.indices()  # 获取稀疏矩阵的索引
    values = A.values() # 获取稀疏矩阵的非零值

    # 将稀疏矩阵转换为密集矩阵
    A_dense = torch.sparse_coo_tensor(indices, values, A.shape).to_dense()

    # 计算度矩阵的逆平方根
    in_degrees = g.in_degrees().float()  # 获取每个节点的入度
    N = torch.diag(1.0 / torch.sqrt(in_degrees))  # 逆平方根
    L = torch.eye(g.number_of_nodes()) - torch.matmul(torch.matmul(N, A_dense), N)  # 计算拉普拉斯矩阵

    # 计算拉普拉斯矩阵的特征值和特征向量
    EigVal, EigVec = np.linalg.eig(L.numpy())
    idx = EigVal.argsort()  # 按特征值排序
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 提取前 pos_enc_dim 个特征向量
    laplacian_feature = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().reshape(-1, pos_enc_dim)
    return laplacian_feature


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

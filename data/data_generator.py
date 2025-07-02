import os
import pickle
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from Bio import PDB
from biopandas.pdb import PandasPdb
from einops import rearrange
from natsort import natsorted
from torch import nn
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
import dgl
import torch
from data.utils import pdb2fasta, laplacian_positional_encoding, \
    ss3_one_hot, sequence_one_hot, pdb2graph_new_chain_info, update_node_feature, update_edge_feature
from data.tri_D import tri_location_D
from data.utils import get_distmaps, process_model, AA_to_tip, edge_positional_embeddings, orientations
from data.dssp import get_dssp


def filter_atoms(test_df: pd.DataFrame, atom_type: str) -> pd.DataFrame:
    if atom_type == 'CB':
        return test_df[((test_df.loc[:, 'residue_name'] == 'GLY') & (test_df.loc[:, 'atom_name'] == 'CA')) \
                       | (test_df.loc[:, 'atom_name'] == 'CB')]
    elif atom_type == 'CA':
        return test_df[test_df.loc[:, 'atom_name'] == 'CA']
    elif atom_type == 'NO':
        return test_df[(test_df.loc[:, 'atom_name'] == 'N') | (test_df.loc[:, 'atom_name'] == 'O')]
    else:
        raise ValueError('Atom type should be CA, CB or NO.')


def distance_helper_v2(pdb_file: str) -> List[np.ndarray]:
    """
    Calculate CA-CA, or CB-CB or N-O distance for a pdb file
    return: distance matrix [CA, CB, NO]
    """
    ppdb = PandasPdb().read_pdb(pdb_file)
    test_df = ppdb.df['ATOM']

    distance_matrix_list = []

    for atom_type in ['CA', 'CB', 'NO']:
        filtered_df = filter_atoms(test_df, atom_type)

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

        distance_matrix_list.append(real_dist)

    return distance_matrix_list


# In: pose, Out: distance maps with different atoms
def extract_multi_distance_map(pdb_file):
    x1 = get_distmaps(pdb_file, atom1="CB", atom2="CB", default="CA")
    x2 = get_distmaps(pdb_file, atom1=AA_to_tip, atom2=AA_to_tip)
    x3 = get_distmaps(pdb_file, atom1="CA", atom2=AA_to_tip)
    x4 = get_distmaps(pdb_file, atom1=AA_to_tip, atom2="CA")
    output = np.stack([x1, x2, x3, x4], axis=-1)
    return output


# tri_d feature
def generate_tri_d(pdb_file):
    maps = extract_multi_distance_map(pdb_file)
    cb_d = maps[:, :, 0]
    L = cb_d.shape[0]
    model_coords, _ = process_model(pdb_file)
    cb_coords = rearrange(
        model_coords,
        "b (l a) d -> b l a d",
        l=L,
    )[0, :, -1, :]
    tri = tri_location_D(cb_d, cb_coords.numpy())
    return tri


def load_esm_embeddings(esm_path, pdb_name):

    esm_file_path = esm_path
    with open(esm_file_path, 'rb') as f:
        esm_data = pickle.load(f)

    return esm_data


def load_pdb_ca_coords(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)
    ca_coordinates = []

    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file {pdb_file_path} not found.")
        return np.array([])

    structure = parser.get_structure(os.path.basename(pdb_file_path), pdb_file_path)

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    ca_coordinates.append(ca_atom.get_coord())

    coords = np.array(ca_coordinates)
    return coords
    

def build_protein_graph(pdb_file: str,
                        fasta_file: str,
                        model_name: str,
                        esm_path: str,
                        out: str,
                        dist_matirx: List[np.ndarray]):
    """Build KNN graph and assign node and edge features. node feature: N * 512, Edge feature: E * 22"""

    print(f'build protein graph step Processing {model_name}')
    scaler = MinMaxScaler()

    # 1. extract sequence from pdb
    sequence = pdb2fasta(pdb_file)

    # 2. build graph and extract edges and PEEE
    _, g, edges, peee = pdb2graph_new_chain_info(pdb_file, knn=10)

    # 3. node features
    # 3.1. dssp
    dssp_feature = get_dssp(fasta_file, pdb_file)
    dssp_feature = torch.cat(dssp_feature, dim=1)  # 按第1维拼接
    dssp_feature = torch.tensor(dssp_feature, dtype=torch.float32)

    # 3.2. sequence one hot as node
    one_hot_feature = sequence_one_hot(sequence)

    # 3.3. laplacian positional encoding
    lap_enc_feature = laplacian_positional_encoding(g, pos_enc_dim=8)

    # 3.4 triangular position feature
    tri = generate_tri_d(pdb_file)
    tri = torch.tensor(tri, dtype=torch.float32)

    # 3.5 esm embedding feature
    esm_file_path = esm_path
    esm_features = []
    if os.path.exists(esm_file_path): 
        esm_embeddings = load_esm_embeddings(esm_path, pdb_file)
        for chain, layers in esm_embeddings.items():
            if isinstance(layers, dict):
                for layer, embedding in layers.items():
                    esm_features.append(np.array(embedding))
            elif isinstance(layers, np.ndarray):
                for i, embedding in enumerate(layers):
                    esm_features.append(np.array(embedding))
            else:
                raise TypeError(f"Unexpected type for layers: {type(layers)}")
    else:
        print(f"Warning: ESM embedding file not found for {pdb_file}")
        return None

    esm_features = np.vstack(esm_features)
    esm_features = torch.tensor(esm_features, dtype=torch.float32)

    # 3.6 coords for input
    pdb_coords = load_pdb_ca_coords(pdb_file)
    pdb_coords = torch.tensor(pdb_coords, dtype=torch.float32)

    # 3.7 ori features
    node_ori = orientations(pdb_coords)
    if isinstance(node_ori, torch.Tensor):
        node_ori_tensor = node_ori.clone().detach()
    else:
        node_ori_tensor = torch.tensor(node_ori, dtype=torch.float32)

    # 4. edge features
    # 4.1. edge sine position encoding
    edge_sin_pos = torch.sin((g.edges()[0] - g.edges()[1]).float()).reshape(-1, 1)

    # 4.2. CA-CA, CB-CB, N-O distance
    # load distance map
    CACA = dist_matirx[0]
    CBCB = dist_matirx[1]
    NO = dist_matirx[2]
    assert CACA.shape == CBCB.shape == NO.shape, f'{model_name} distance map shape not match: {CACA.shape}, {CBCB.shape}, {NO.shape}'

    caca_feature, cbcb_feature, no_feature = [], [], []

    for i in edges:
        caca_feature.append(CACA[i])
        cbcb_feature.append(CBCB[i])
        no_feature.append(NO[i])

    contact_feature = torch.tensor([1 if cb_distance < 8.0 else 0 for cb_distance in cbcb_feature]).reshape(-1, 1)
    caca_feature = torch.tensor(scaler.fit_transform(torch.tensor(caca_feature).reshape(-1, 1)))
    cbcb_feature = torch.tensor(scaler.fit_transform(torch.tensor(cbcb_feature).reshape(-1, 1)))
    no_feature = torch.tensor(scaler.fit_transform(torch.tensor(no_feature).reshape(-1, 1)))

    # 4.3 egde position embedding
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0)
    edge_pos_encodings = edge_positional_embeddings(g, edge_index, num_embeddings=16)

    # 5. add feature to graph
    update_node_feature(g, [dssp_feature, one_hot_feature, lap_enc_feature, tri])

    g.ndata['coords'] = pdb_coords
    g.ndata['ori'] = node_ori_tensor
    g.ndata['esm'] = esm_features

    update_edge_feature(g, [edge_sin_pos, caca_feature,
                            cbcb_feature, no_feature,
                            contact_feature, peee, edge_pos_encodings])

    dgl.save_graphs(filename=out, g_list=g)
    return None


def wrapper(pdb_file: str, fasta_file: str, esm_file: str, dgl_file: str, pdb_id: str):
    print(f"wrapper step Processing PDB file: {pdb_file}")
    dist_matirx_list = distance_helper_v2(pdb_file)

    build_protein_graph(pdb_file=pdb_file,
                        fasta_file=fasta_file,
                        model_name=pdb_id,
                        esm_path=esm_file,
                        out=dgl_file,
                        dist_matirx=dist_matirx_list)

    return None


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate protein complex structures')
    parser.add_argument('--input_pdb_folder', '-i', type=str, help='pdb file path', required=True)
    parser.add_argument('--fasta_folder', '-f', type=str, help='fasta file path', required=True)
    parser.add_argument('--dgl_save_folder', '-o', type=str, help='output folder', required=True)
    parser.add_argument('--esm_pkl_folder', '-e', type=str, help='esm file path', required=True)
    parser.add_argument('--cores', '-c', type=int, help='multi-cores', required=False, default=1)
    args = parser.parse_args()

    input_pdb_folder = args.input_pdb_folder
    dgl_save_folder = args.dgl_save_folder
    cores = args.cores

    if not os.path.isdir(input_pdb_folder):
        raise FileNotFoundError(f'Please check input pdb folder {input_pdb_folder}')
    input_pdb_folder = os.path.abspath(input_pdb_folder)  # get absolute path

    subfolders = natsorted([f.path for f in os.scandir(input_pdb_folder) if f.is_dir()])

    for subfolder in subfolders:
        output_folder = os.path.join(subfolder, 'output')
        if os.path.isdir(output_folder):
            pdbs = natsorted(
                [os.path.join(output_folder, pdb) for pdb in os.listdir(output_folder) if pdb.endswith('.pdb')])
            if len(pdbs) == 0:
                print(f"No PDB files found in {output_folder}")
                continue
            fasta_folder = os.path.join(args.fasta_folder, f'{os.path.basename(subfolder)}')

            if not os.path.exists(fasta_folder):
                print(f"Warning: fasta folder not found for {fasta_folder}")
                continue

            esm_folder = os.path.join(args.esm_pkl_folder, f'{os.path.basename(subfolder)}_embedding')
            if not os.path.exists(esm_folder):
                print(f"Warning: ESM embedding folder not found for {esm_folder}")
                continue

            dgl_subfolder = os.path.join(dgl_save_folder, os.path.basename(subfolder))
            os.makedirs(dgl_subfolder, exist_ok=True)

            for pdb_file in tqdm(pdbs):
                pdb_base_name = os.path.basename(pdb_file).replace('.pdb', '')

                fasta_file = os.path.join(fasta_folder, f"{pdb_base_name}.fasta")
                if not os.path.isfile(fasta_file):
                    print(f"Warning: fasta file not found for {pdb_file}")
                    continue

                esm_pkl_file = os.path.join(esm_folder, f"{pdb_base_name}_emb.pkl")

                if not os.path.isfile(esm_pkl_file):
                    print(f"Warning: ESM embedding file not found for {pdb_file}")
                    continue

                dgl_file_name = f"{pdb_base_name}.dgl"
                dgl_save_path = os.path.join(dgl_subfolder, dgl_file_name)

                wrapper(pdb_file, fasta_file, esm_pkl_file, dgl_save_path, pdb_base_name)

    print('All done.')

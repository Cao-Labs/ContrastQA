"""
It uses for testing.
"""

import os
from tqdm import tqdm
import torch
from natsort import natsorted
from torch.utils.data import DataLoader
from data.data_generator import distance_helper_v2, build_protein_graph
from data.utils import pdb2fasta
from model.dataset import TestData, collate
from model.test_model import QAModel
import re
import sys
from os.path import join, isdir, isfile
from timeit import default_timer as timer

import numpy as np
import warnings
import pandas as pd
from esm_main.generate_embedding.generate_embed_test import generate_embed_pt
from esm_main.generate_embedding.save_emb_pkl_test import save_esm_emb_pkl

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PYTHON_INSTALL = 'python3'


def save_fasta_from_pdb(pathToInput, pathToSave):
    """Convert all PDB files in a directory to FASTA files in another directory"""
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    fasta_dir = join(pathToTempDirectory, 'fasta')
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)  # Create output directory if it doesn't exist

    for filename in os.listdir(pathToInput):
        if filename.endswith(".pdb"):
            pdb_file = os.path.join(pathToInput, filename)
            seq = pdb2fasta(pdb_file)  # Extract sequence from PDB file

            # Generate corresponding FASTA file name
            fasta_filename = f"{filename.split('.')[0]}.fasta"
            fasta_file = os.path.join(fasta_dir, fasta_filename)

            # Write sequence to the FASTA file
            with open(fasta_file, 'w') as f:
                f.write(f">{filename.split('.')[0]}\n")  # Header with PDB ID
                for i in range(0, len(seq), 60):  # Split sequence into lines of 60 characters
                    f.write(seq[i:i + 60] + "\n")

    return fasta_dir


def wrapper(pdb_file: str, fasta_file: str, esm_file: str, dgl_file: str, pdb_id: str):
    try:
        if os.path.exists(dgl_file):
            return None

        dist_matirx_list = distance_helper_v2(pdb_file)

        # build protein graph
        build_protein_graph(pdb_file=pdb_file,
                            fasta_file=fasta_file,
                            model_name=pdb_id,
                            esm_path=esm_file,
                            out=dgl_file,
                            dist_matirx=dist_matirx_list)

        print(f"Processed {pdb_id} and saved to {dgl_file}.")
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
    return None


def esm_emb_generate(pathToInput, pathToSave):
    
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    model = os.path.join(".", "esm_main", "esm_pt", "esm2_t33_650M_UR50D.pt")
    esm_pt_path = join(pathToTempDirectory, 'esm_data')
    esm_pkl_path = join(pathToTempDirectory, 'esm_pkl')

    if not os.path.exists(esm_pt_path):
        os.makedirs(esm_pt_path)

    if os.listdir(esm_pt_path):
        print(f"Skipping ESM embedding generation for {pathToInput}: output directory already contains files.")
    else:
        generate_embed_pt(pathToInput, model, esm_pt_path)
        print(f"Generating ESM embeddings and saving to {esm_pt_path}")

    if not os.path.exists(esm_pkl_path):
        os.makedirs(esm_pkl_path)

    print("Converting .pt files to .pkl...")
    if os.listdir(esm_pkl_path):
        print(f'Skipping ESM embedding pkl generation for {esm_pt_path}: output directory already contains files. ')
    else:
        save_esm_emb_pkl(esm_pt_path, esm_pkl_path)

    print(f"ESM embeddings have been successfully saved in {esm_pkl_path}")

    return esm_pkl_path


def dgl_generate(pdb_data_path, fasta_data_path, pathToSave, esm_data_path):
    """
    Args:
        pdb_data_path: PDB file input path
        pathToSave: dgl graph save path
        esm_data_path:esm data file input path
    Returns: dgl file
    """
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    dgl_generate_path = join(pathToTempDirectory, 'dgl_generate')

    if not os.path.isdir(pdb_data_path):
        raise FileNotFoundError(f'Please check input pdb folder {pdb_data_path}')
    pdb_data_path = os.path.abspath(pdb_data_path)  # get absolute path

    if not os.path.exists(dgl_generate_path):
        os.makedirs(dgl_generate_path)
        
    if os.path.isdir(pdb_data_path):
        pdbs = natsorted([os.path.join(pdb_data_path, pdb) for pdb in os.listdir(pdb_data_path) if pdb.endswith('.pdb')])

        if len(pdbs) == 0:
            print(f"No PDB files found in {pdb_data_path}")

        esm_folder = os.path.join(esm_data_path)
        if not os.path.exists(esm_folder):
            print(f"Warning: ESM embedding folder not found for {esm_folder}")

        for pdb_file in tqdm(pdbs):
            pdb_base_name = os.path.basename(pdb_file).replace('.pdb', '')

            fasta_file = os.path.join(fasta_data_path, f"{pdb_base_name}.fasta")

            esm_pkl_file = os.path.join(esm_folder, f"{pdb_base_name}_emb.pkl")

            if not os.path.isfile(esm_pkl_file):
                print(f"Warning: ESM embedding file not found for {pdb_file}")
                continue

            dgl_file_name = f"{pdb_base_name}.dgl"
            dgl_save_path = os.path.join(dgl_generate_path, dgl_file_name)

            if os.path.exists(dgl_save_path):
                print(f"File {dgl_save_path} already exists, skipping...")
                continue

            wrapper(pdb_file, fasta_file, esm_pkl_file, dgl_save_path, pdb_base_name)

    print('DGL generate All done.')
    return dgl_generate_path


def evaluate_QA_results(dgl_input, pathToSave):
    dgl_folder = dgl_input
    result_folder = pathToSave

    eval_set = TestData(dgl_folder)
    eval_loader = DataLoader(eval_set.data,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    model_list = os.listdir(dgl_folder)
    model_list = [i.split('.')[0] for i in model_list]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_file = f'model/checkpoints/GVP_GCL_temperature_0.2/epoch=45-val_loss=0.09626.ckpt'
    model = QAModel.load_from_checkpoint(ckpt_file)
    print(f'Loading {ckpt_file}')
    model = model.to(device)
    model.eval()  # turn on model eval mode

    per_target_models_stat = pd.DataFrame(
        columns=['target', 'model', 'pred_score'])

    eval_num = 0
    for idx, batch_graphs in enumerate(eval_loader):

        batch_graphs = batch_graphs.to(device)

        # generate testing scores
        batch_scores, batch_class = model.forward(batch_graphs)
        batch_scores = torch.clamp(batch_scores, min=0, max=1)

        for model_idx in range(batch_scores.size(0)):
            eval_num += 1
            model_name = model_list[eval_num - 1]  

            # extract target_name
            if model_name.startswith("fold_"):  # process like "fold_7o9w_1_model_0" abag-af3 dataset
                match = re.search(r"fold_(\w+?)_", model_name)
                if match:
                    target_name = match.group(1)
                else:
                    target_name = model_name  
            else:
                target_name = model_name[:5]  # process like "H1106" CASP16 dataset

            node_prob = batch_scores[model_idx].cpu().data.numpy().flatten()
            summary_pred = pd.DataFrame({'Pred': node_prob}, columns=['Pred'])

            # calculate global lddt score
            model_pred_lddt_global = np.mean(summary_pred['Pred'])

            # product DataFrame
            new_row = pd.DataFrame([{
                'target': target_name,
                'model': model_name,
                'pred_score': model_pred_lddt_global
            }])

            per_target_models_stat = pd.concat([per_target_models_stat, new_row], ignore_index=True)

    per_target_models_stat['pred_score'] = per_target_models_stat['pred_score'].astype(float)

    # sorting each target，and save as CSV file
    for targetid in per_target_models_stat['target'].unique():
        data_subset = per_target_models_stat[per_target_models_stat['target'] == targetid].reset_index()

        data_subset.sort_values(by='pred_score', ascending=False, inplace=True)
        data_subset.loc[:, 'pred_score'] = data_subset.loc[:, 'pred_score'].round(5)

        data_subset.to_csv(os.path.join(result_folder, f'Rank_{targetid}_qa.csv'), index=False)

    print(f"Results saved to {result_folder}")

def main(pathToInput, pathToSave):
    start = timer()

    pattern = re.compile(r'[a-zA-Z0-9]{4,20}')
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    create_folder(pathToSave)

    # generate fasta file
    fasta_data_path = save_fasta_from_pdb(pathToInput, pathToSave)

    # generate esm pkl file
    esm_input_path = esm_emb_generate(pathToInput, pathToSave)
    print('esm data generated...')

    if not os.path.exists(esm_input_path) or not os.listdir(esm_input_path):
        print(f"Error: {esm_input_path} does not exist or is empty. Ensure the pkl files are correctly generated.")
        sys.exit()

    # pdb file
    pdb_input_path = pathToInput

    # transform pkl to dgl file
    dgl_input_path = dgl_generate(pdb_input_path, fasta_data_path, pathToSave, esm_input_path)
    print('dgl graphs generation done...')

    # result
    evaluate_QA_results(dgl_input_path, pathToSave)

    print(f"Prediction saved to {pathToSave}")
    print("Cleaning up...")

    # delete tmp files
    folder_to_remove = join(pathToSave, 'tmp')
    os.system(f'rm -rf {folder_to_remove}')
    end = timer()
    total_t = end - start
    print(f"Prediction complete, elapsed time: {total_t}")

def create_folder(pathToFolder):
    """
    Method to create folder if does not exist, pass if it does exist,
    cancel the program if there is another error (e.g writing to protected directory)
    """
    try:
        os.mkdir(pathToFolder)
    except FileExistsError:
        print(f"{pathToFolder} already exists...")
    except:
        print(f"Fatal error making {pathToFolder}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Not enough arguments... example command: ')
        print(f'python {sys.argv[0]} /path/To/Input/folder/ /path/to/output/save')
        sys.exit()

    pathToInput = sys.argv[1]
    pathToSave = sys.argv[2]

    main(pathToInput, pathToSave)

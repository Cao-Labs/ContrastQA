"""
It uses for dproq feature & tri & ESM2 embedding
"""

import os
from tqdm import tqdm
import torch
from natsort import natsorted
from torch.utils.data import DataLoader
# 获取当前文件的绝对路径和父目录
from data.dproq_feature.data_generator_v3 import distance_helper_v2, build_protein_graph
from data.dproq_feature.utils import pdb2fasta
from model.dataset import TestData, collate
from model.network_gvp_gcl_lddt_v1_dproq_test import QAModel
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
        # 如果目标 DGL 文件已经存在，则跳过处理
        if os.path.exists(dgl_file):
            return None

        dist_matirx_list = distance_helper_v2(pdb_file)

        # 构建蛋白质图并保存为 DGL 文件
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
    """
    Args:
        pathToInput: 输入的PDB文件路径
        pathToSave: ESM嵌入特征pkl文件保存路径

    Returns: pkl格式文件
    """
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    model = os.path.join(".", "esm_main", "esm_pt", "esm2_t33_650M_UR50D.pt")
    esm_pt_path = join(pathToTempDirectory, 'esm_data')
    esm_pkl_path = join(pathToTempDirectory, 'esm_pkl')

    if not os.path.exists(esm_pt_path):
        os.makedirs(esm_pt_path)

    # 在调用前添加检查
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
        pdb_data_path: PDB文件输入路径
        pathToSave: dgl图保存路径
        esm_data_path:预处理的ESM嵌入特征pkl文件输入路径
        cores:多进程数
    Returns: dgl格式文件
    :param esm_data_path:
    :param pdb_data_path:
    :param fasta_data_path:
    """
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    dgl_generate_path = join(pathToTempDirectory, 'dgl_generate')

    if not os.path.isdir(pdb_data_path):
        raise FileNotFoundError(f'Please check input pdb folder {pdb_data_path}')
    pdb_data_path = os.path.abspath(pdb_data_path)  # get absolute path

    # 确保输出目录存在
    if not os.path.exists(dgl_generate_path):
        os.makedirs(dgl_generate_path)

    # 遍历所有PDB文件
    if os.path.isdir(pdb_data_path):
        pdbs = natsorted([os.path.join(pdb_data_path, pdb) for pdb in os.listdir(pdb_data_path) if pdb.endswith('.pdb')])

        if len(pdbs) == 0:
            print(f"No PDB files found in {pdb_data_path}")

        # 获取对应的 esm 文件路径
        esm_folder = os.path.join(esm_data_path)
        # 确保 ESM embedding 文件夹存在
        if not os.path.exists(esm_folder):
            print(f"Warning: ESM embedding folder not found for {esm_folder}")

        # 处理每个 PDB 文件
        for pdb_file in tqdm(pdbs):
            # 获取 PDB 文件的基名（不带后缀）
            pdb_base_name = os.path.basename(pdb_file).replace('.pdb', '')

            fasta_file = os.path.join(fasta_data_path, f"{pdb_base_name}.fasta")

            # 对应的 ESM embedding 文件名
            esm_pkl_file = os.path.join(esm_folder, f"{pdb_base_name}_emb.pkl")

            if not os.path.isfile(esm_pkl_file):
                print(f"Warning: ESM embedding file not found for {pdb_file}")
                continue

            # 生成 DGL 文件保存路径
            dgl_file_name = f"{pdb_base_name}.dgl"
            dgl_save_path = os.path.join(dgl_generate_path, dgl_file_name)

            # 如果文件已存在，则跳过
            if os.path.exists(dgl_save_path):
                print(f"File {dgl_save_path} already exists, skipping...")
                continue

            # 如果 ESM 文件存在，则继续处理
            wrapper(pdb_file, fasta_file, esm_pkl_file, dgl_save_path, pdb_base_name)

    print('DGL generate All done.')
    # 返回 dgl 生成目录路径
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

    # 获取所有模型的文件名，并去掉扩展名，生成模型名列表
    model_list = os.listdir(dgl_folder)
    model_list = [i.split('.')[0] for i in model_list]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_file = f'model/checkpoints/GVP_GCL_temperature_0.2/epoch=45-val_loss=0.09626.ckpt'
    model = QAModel.load_from_checkpoint(ckpt_file)
    print(f'Loading {ckpt_file}')
    model = model.to(device)
    model.eval()  # turn on model eval mode

    # 初始化一个空的 DataFrame
    per_target_models_stat = pd.DataFrame(
        columns=['target', 'model', 'pred_score'])

    eval_num = 0
    for idx, batch_graphs in enumerate(eval_loader):

        batch_graphs = batch_graphs.to(device)

        # 调用模型的 forward 方法，获取预测结果
        batch_scores, batch_class = model.forward(batch_graphs)
        batch_scores = torch.clamp(batch_scores, min=0, max=1)

        # 遍历 batch 中的每个图，分别计算全局和局部 lDDT
        for model_idx in range(batch_scores.size(0)):
            eval_num += 1
            model_name = model_list[eval_num - 1]  # 根据 eval_num 从 model_list 中获取模型名

            # 提取 target_name
            if model_name.startswith("fold_"):  # 处理像 fold_7o9w_1_model_0 的情况
                match = re.search(r"fold_(\w+?)_", model_name)
                if match:
                    target_name = match.group(1)
                else:
                    target_name = model_name  # fallback: 原样返回
            elif model_name.startswith("7"):  # 处理像 fold_7o9w_1_model_0 的情况
                target_name = model_name[:4]
            else:
                target_name = model_name[:5]  # 默认前5位，如 H1106

            # 提取预测的节点分数并 reshape
            node_prob = batch_scores[model_idx].cpu().data.numpy().flatten()
            summary_pred = pd.DataFrame({'Pred': node_prob}, columns=['Pred'])

            # 计算全局 lDDT 分数
            model_pred_lddt_global = np.mean(summary_pred['Pred'])

            # 过滤掉值为 -1 的残基（即缺失的残基）
            model_pred_lddt_local = summary_pred['Pred'].to_list()
            model_pred_lddt_local = [round(x, 5) for x in model_pred_lddt_local]

            # 将新的数据行构造为一个 DataFrame
            new_row = pd.DataFrame([{
                'target': target_name,
                'model': model_name,
                'pred_score': model_pred_lddt_global
            }])

            # 使用 pd.concat 代替 append，将新行添加到 per_target_models_stat 中
            per_target_models_stat = pd.concat([per_target_models_stat, new_row], ignore_index=True)

    per_target_models_stat['pred_score'] = per_target_models_stat['pred_score'].astype(float)

    # 对每个 target 进行排序，并保存结果为 CSV 文件
    for targetid in per_target_models_stat['target'].unique():
        data_subset = per_target_models_stat[per_target_models_stat['target'] == targetid].reset_index()

        data_subset.sort_values(by='pred_score', ascending=False, inplace=True)
        data_subset.loc[:, 'pred_score'] = data_subset.loc[:, 'pred_score'].round(5)

        # 保存每个 target 的结果为单独的 CSV 文件
        data_subset.to_csv(os.path.join(result_folder, f'Rank_{targetid}_qa_gvp_gcl_temperature0.2.csv'), index=False)

    print(f"Results saved to {result_folder}")


# 用于将输入数据处理成模型所需的格式，生成预测结果，并保存这些结果。最后还会清理临时文件夹并打印总耗时
# pathToInput: 字符串，表示输入数据的路径。
# pathToSave: 字符串，表示保存预测结果的路径
def main(pathToInput, pathToSave):
    start = timer()

    # 正则表达式匹配目标名称，如果未找到则默认设置为 'Target'
    pattern = re.compile(r'[a-zA-Z0-9]{4,20}')
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    # 创建保存路径
    create_folder(pathToSave)

    # 生成 fasta 文件
    fasta_data_path = save_fasta_from_pdb(pathToInput, pathToSave)

    # 生成esm嵌入特征pkl文件
    esm_input_path = esm_emb_generate(pathToInput, pathToSave)
    print('esm data generated...')

    # 检查是否生成了 esm pkl 文件
    if not os.path.exists(esm_input_path) or not os.listdir(esm_input_path):
        print(f"Error: {esm_input_path} does not exist or is empty. Ensure the pkl files are correctly generated.")
        sys.exit()

    # pdb文件
    pdb_input_path = pathToInput

    # 将 pkl 文件转换为 DGL 文件
    dgl_input_path = dgl_generate(pdb_input_path, fasta_data_path, pathToSave, esm_input_path)
    print('dgl graphs generation done...')

    # 评估 QA 结果
    evaluate_QA_results(dgl_input_path, pathToSave)

    print(f"Prediction saved to {pathToSave}")
    print("Cleaning up...")

    # 删除临时文件夹
    # folder_to_remove = join(pathToSave, 'tmp')
    # os.system(f'rm -rf {folder_to_remove}')
    end = timer()
    total_t = end - start
    print(f"Prediction complete, elapsed time: {total_t}")


# 用于在指定路径创建文件夹。如果文件夹已经存在则跳过创建操作，如果出现其他错误则打印错误消息
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


# 定义了一个脚本的入口点，它检查命令行参数的数量，如果参数不足则提示用户正确的命令格式并退出程序。
# 如果参数数量足够，则调用 main 函数执行主要逻辑
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Not enough arguments... example command: ')
        print(f'python {sys.argv[0]} /path/To/Input/folder/ /path/to/output/save')
        sys.exit()

    pathToInput = sys.argv[1]
    pathToSave = sys.argv[2]

    main(pathToInput, pathToSave)

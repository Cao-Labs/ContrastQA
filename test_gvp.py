import os
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 获取当前文件的绝对路径和父目录
from model.dataset import TestData, collate
from model.network_gvp_lddt_v2_1 import QAModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import sys
import subprocess
from os.path import join, isdir, isfile
from timeit import default_timer as timer

import numpy as np
import warnings

from script.paths import PATHS
from model.dataload import ProteinGVPQAData_Generator, load_esm_embeddings
import tensorflow as tf
import pandas as pd

from esm_main.generate_embedding.generate_embed_test import generate_embed_pt
from esm_main.generate_embedding.save_emb_pkl_test import save_esm_emb_pkl

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CUDA_LAUNCH_BLOCKING = 1
PYTHON_INSTALL = 'python3'

#  GCN 所需的参数
node_input_dim = 300
hidden_dim = 128
out_dim = 32
feature_dim = 300
num_layers = 2
read_out = 'sum'
init_lr = 0.0005
weight_decay = 5e-4
opt = 'adam'


# 负责预处理 PDB 输入文件，并将必要的特征提取到 pickle 文件中，以便模型输入时使用。
# 它将中间步骤保存到一个临时目录，并返回最终步骤的输出路径供下一个步骤使用
def preprocess_input(pathToInput, pathToSave):
    """
    This method is responsible for taking the pdb input files and extract
    all of the necesary features into the pickle files that can be easily
    transformed into the correct format for the model input, saves it to a temp
    directory in the output folder

    预处理 PDB 输入文件，并将必要的特征提取到 pickle 文件中，以便模型输入时使用。
    它将中间步骤保存到一个临时目录，并返回最终步骤的输出路径供下一个步骤使用。

    Parameters:
    ----------------
    pathToInput: string
        This is a string representation to the path to the input data

    pathToSave: string
        This is a string representation to the path to the save folder, so we can
        make a temp folder to store the intermediary steps

    Return:
    ---------------
    type: string
        The return is the path to the final step (step2_generate_casp_fragment_structures)
        output so that the next step can use it as the input

    """
    # 使用正则表达式从 pathToInput 中提取目标名称，如果未找到则默认设置为 'Target'
    pattern = re.compile(r'[a-zA-Z0-9]{4,7}')
    target_name = re.search(pattern, pathToInput)

    if target_name is not None:
        target_name = str(target_name[0])
    else:
        target_name = 'Target'

    # 创建多个目录路径以存储中间数据，包括清理后的 PDB 文件路径、步骤0的输出路径、JSON数据路径和最终输出路径
    print("Created tmp directory...")
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    clean_data_path = join(pathToTempDirectory, 'cleaned_pdbs')
    pathToStep0 = join(pathToTempDirectory, 'step_0')
    pathToJSON = join(pathToTempDirectory, 'JSON_Data')
    pathToGVPQAInputData = join(pathToTempDirectory, 'GCN_feature')

    # 创建清理后的 PDB 文件夹，并使用 Perl 脚本重新编号残基索引，清理每个 PDB 文件
    print("Processing input data...")
    create_folder(clean_data_path)
    clean_data_program = join(PATHS.sw_install, 'script/re_number_residue_index.pl')
    clean_data_command = "perl {} {} {}"
    for pdb in os.listdir(pathToInput):
        command = clean_data_command.format(clean_data_program, join(pathToInput, pdb), join(clean_data_path, pdb))
        subprocess.run(command.split(" "))
    print("Cleaned PDB's")

    # 创建步骤0输出文件夹，使用 Python 脚本在内部调用 Perl 脚本为每个 PDB 文件添加链信息
    print("Creating Step 0 output folder...")
    create_folder(pathToStep0)
    pathToStep0OUT = join(pathToStep0, target_name)
    step0_location = join(PATHS.sw_install, 'script/step0_prepare_add_chain_to_folder.py')
    chain_add_location = join(PATHS.sw_install, 'script/assist_add_chainID_to_one_pdb.pl')
    chain_add_command = f'{PYTHON_INSTALL} {step0_location} {chain_add_location} {clean_data_path} {pathToStep0OUT}'
    subprocess.run(chain_add_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('1/3 done...')

    # 使用 Python 脚本从 PDB 文件提取结构信息,并将其转换为更易于处理的JSON数据，保存到 pathToJSON
    print("Creating JSON data...")
    create_folder(pathToJSON)
    step1_location = join(PATHS.sw_install, 'script/step1_create_json_from_PDB.py')
    stride_location = join(PATHS.sw_install, 'script/stride_bin/stride')
    json_command = f'{PYTHON_INSTALL} {step1_location} {stride_location} {pathToStep0} {pathToJSON}'
    subprocess.run(json_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('2/3 done...')

    # 检查目标目录中是否已有文件存在
    if os.path.exists(pathToGVPQAInputData) and os.listdir(pathToGVPQAInputData):
        print("PKL files already exist in the target directory. Skipping generation step...")
        return pathToGVPQAInputData

    # 使用 Python 脚本生成片段结构，保存到 pathToVGAEQAInputData
    print("Starting generate PKL files...")
    step2_location = join(PATHS.sw_install, 'script/step2_generate_fragment_structures_v2.py')
    rfpredictions_locations = join(PATHS.sw_install, 'script/assist_generation_scripts/RF_Predictions/')
    frag_structure_command = f'{PYTHON_INSTALL} {step2_location} {pathToJSON} {rfpredictions_locations} ' \
                             f'{pathToGVPQAInputData} > {join(pathToTempDirectory, "step2_log.txt")} 2>&1'

    # 创建输出目录
    create_folder(pathToGVPQAInputData)

    # 运行命令并检查日志文件
    subprocess.run(frag_structure_command, shell=True)
    print('3/3 done...')

    # 检查生成的 pkl 文件
    if not os.path.exists(pathToGVPQAInputData) or not os.listdir(pathToGVPQAInputData):
        print(
            f"Error: {pathToGVPQAInputData} does not exist or is empty. Ensure the pkl files are correctly generated.")
        sys.exit()

    print("pkl files generated successfully...")
    return pathToGVPQAInputData


def esm_emb_generate(pathToInput,  pathToSave):
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


def dgl_generate(processed_data_path, esm_data_path, pathToSave):
    """
    Args:
        processed_data_path: 预处理的pkl文件输入路径
        esm_data_path:预处理的ESM嵌入特征pkl文件输入路径
        pathToSave: dgl图保存路径

    Returns: dgl格式文件
    """
    pathToTempDirectory = join(pathToSave, 'tmp')
    create_folder(pathToTempDirectory)
    dgl_generate_path = join(pathToTempDirectory, 'dgl_generate')

    # 创建数据生成器实例
    data_gen = ProteinGVPQAData_Generator(
        data_path=processed_data_path,
        esm_path=esm_data_path,
        protein_info_pickle='./data/test_info/T1109o_info.csv',
        min_seq_size=0,
        max_seq_size=10000,
        batch_size=1,
        max_msa_seq=100000,
        max_id_nums=1000000
    )

    # 获取数据集
    dataset = data_gen.get_dataset()

    # 确保输出目录存在
    if not os.path.exists(dgl_generate_path):
        os.makedirs(dgl_generate_path)

    # 获取所有 pkl 文件并按自然数顺序排序
    def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

    pkl_files = [f for f in os.listdir(processed_data_path) if f.endswith('.pkl')]
    esm_files = [f for f in os.listdir(esm_data_path) if f.endswith('_emb.pkl')]  # 假设 ESM 文件命名规则
    pkl_files.sort(key=natural_sort_key)
    esm_files.sort(key=natural_sort_key)

    # 迭代数据集并转换为 DGL 图并保存
    for idx, (node_features, contact_map, _) in enumerate(dataset):
        print(f"Dataset {idx}: node_features shape: {node_features.shape}")
        if idx >= len(pkl_files) or idx >= len(esm_files):
            break

        pkl_filename = pkl_files[idx]
        esm_filename = esm_files[idx]
        pdb_index = os.path.splitext(pkl_filename)[0]
        dgl_filename = pdb_index + '.dgl'
        filepath = os.path.join(dgl_generate_path, dgl_filename)

        # 如果文件已存在，则跳过
        if os.path.exists(filepath):
            print(f"File {filepath} already exists, skipping...")
            continue

        # 转换为 DGL 图
        dgl_graphs = data_gen.convert_to_dgl(node_features.numpy(), contact_map.numpy())

        if dgl_graphs:
            data_gen.save_dgl_graphs(dgl_graphs, filepath)
            print(f"Processed {pkl_filename} and {esm_filename} -> {dgl_filename}")
        else:
            print(f"Warning: No graphs generated for {pkl_filename}")

    # 返回 dgl 生成目录路径
    return dgl_generate_path


def load_pdb_coords(pdb_file):
    # 创建 PDB 解析器
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB_structure', pdb_file)

    # 存储坐标
    coordinates = []

    # 遍历每个模型、链和残基
    for model in structure:
        for chain in model:
            for residue in chain:
                # 提取 CA 原子的坐标
                if residue.has_id('CA'):
                    coordinates.append(residue['CA'].get_coord())

    # 转换为 NumPy 数组
    coords = np.array(coordinates)

    return coords


def evaluate_QA_results(dgl_input, pdb_input, pathToSave):
    dgl_folder = dgl_input
    result_folder = pathToSave

    eval_set = TestData(dgl_folder)
    eval_loader = DataLoader(eval_set.data,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True,
                             collate_fn=collate,
                             shuffle=False)

    # 获取所有模型的文件名，并去掉扩展名，生成模型名列表
    model_list = os.listdir(dgl_folder)
    model_list = [i.split('.')[0] for i in model_list]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ckpt_file = f'model/checkpoints/GVP_lddt_v1/epoch=30-val_loss=0.01804.ckpt'
    model = QAModel.load_from_checkpoint(ckpt_file)
    print(f'Loading {ckpt_file}')
    model = model.to(device)
    model.eval()

    # 初始化一个空的 DataFrame
    per_target_models_stat = pd.DataFrame(
        columns=['target', 'model', 'interface_local_score', 'interface_score'])

    eval_num = 0
    for idx, batch_graphs in enumerate(eval_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(torch.float).to(device)
        batch_x = batch_x.squeeze()
        batch_e = batch_graphs.edata['weight'].to(torch.float).to(device)
        batch_e = batch_e.squeeze()
        batch_e = batch_e.unsqueeze(dim=1)
        # 提取当前批次 DGL 文件名
        batch_dgl_files = eval_set.file_names[idx * 1: (idx + 1) * 1]  # 假设 eval_set 中有一个 file_names 列表，保存了文件名

        # 根据 DGL 文件名在 PDB 文件夹中找到对应的 PDB 文件
        batch_pdb_files = []
        for dgl_file in batch_dgl_files:
            pdb_file_name = os.path.join(pdb_input, os.path.splitext(dgl_file)[0] + '.pdb')
            if os.path.exists(pdb_file_name):
                batch_pdb_files.append(pdb_file_name)
            else:
                raise FileNotFoundError(f"PDB file {pdb_file_name} not found.")

        # 假设 batch_coords 是一个列表，包含每个 PDB 文件的坐标张量
        batch_coords = [load_pdb_coords(pdb_file) for pdb_file in batch_pdb_files]

        # 将每个坐标转换为张量并放入列表中
        batch_coords_tensors = [torch.tensor(coords, dtype=torch.float32) for coords in batch_coords]

        # 使用 torch.cat 拼接所有张量
        batched_coords = torch.cat(batch_coords_tensors, dim=0).to(device)

        # 如果需要，可以使用 squeeze() 来去掉多余的维度
        batch_coords = batched_coords.squeeze()

        # 提取源节点索引 (src_index) 和目标节点索引 (dst_index)
        src_index, dst_index = batch_graphs.edges()

        # 分别对 src_index 和 dst_index 进行 squeeze 操作
        src_index = src_index.squeeze()
        dst_index = dst_index.squeeze()

        # 将 src_index 和 dst_index 堆叠成形状为 [2, num_edges] 的张量
        edge_index = torch.stack([src_index, dst_index], dim=0)

        # 调用模型的 forward 方法，获取预测结果
        batch_scores = model.forward(batch_graphs, batch_x, batch_coords, edge_index, batch_e,
                                     scatter_mean=True, dense=True)
        batch_scores = torch.clamp(batch_scores, min=0, max=1)

        if batch_scores.dim() == 0:
            batch_scores = batch_scores.unsqueeze(0)  # 将其转换为一维张量

        # 遍历 batch 中的每个图，分别计算全局和局部 lDDT
        for model_idx in range(batch_scores.size(0)):
            eval_num += 1
            model_name = model_list[eval_num - 1]  # 根据 eval_num 从 model_list 中获取模型名

            # 提取 target 名称（如 H1106），即模型名的前 5 个字符
            target_name = model_name[:5]  # 假设 target_name 是模型名的前 5 个字符，如 H1106

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
                'interface_local_score': model_pred_lddt_local,
                'interface_score': model_pred_lddt_global
            }])

            # 使用 pd.concat 代替 append，将新行添加到 per_target_models_stat 中
            per_target_models_stat = pd.concat([per_target_models_stat, new_row], ignore_index=True)

    per_target_models_stat['interface_score'] = per_target_models_stat['interface_score'].astype(float)

    # 对每个 target 进行排序，并保存结果为 CSV 文件
    for targetid in per_target_models_stat['target'].unique():
        data_subset = per_target_models_stat[per_target_models_stat['target'] == targetid].reset_index()

        data_subset.sort_values(by='interface_score', ascending=False, inplace=True)
        data_subset.loc[:, 'interface_score'] = data_subset.loc[:, 'interface_score'].round(5)

        # 保存每个 target 的结果为单独的 CSV 文件
        data_subset.to_csv(os.path.join(result_folder, f'Rank_{targetid}_complexqa.csv'), index=False)

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

    # 预处理输入数据并生成 pkl 文件
    model_input_path = preprocess_input(pathToInput, pathToSave)
    print("Input data created...")

    # 检查是否生成了 pkl 文件
    if not os.path.exists(model_input_path) or not os.listdir(model_input_path):
        print(f"Error: {model_input_path} does not exist or is empty. Ensure the pkl files are correctly generated.")
        sys.exit()

    # 生成esm嵌入特征pkl文件
    esm_input_path = esm_emb_generate(pathToInput, pathToSave)
    print('esm data generated...')

    # 检查是否生成了 esm pkl 文件
    if not os.path.exists(esm_input_path) or not os.listdir(esm_input_path):
        print(f"Error: {esm_input_path} does not exist or is empty. Ensure the pkl files are correctly generated.")
        sys.exit()

    # 将 pkl 文件转换为 DGL 文件
    dgl_input_path = dgl_generate(model_input_path, esm_input_path, pathToSave)
    print('dgl graphs generation done...')

    # pdb文件
    pdb_input_path = pathToInput

    # 评估 QA 结果
    evaluate_QA_results(dgl_input_path, pdb_input_path, pathToSave)

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
    '''
    Method to create folder if does not exist, pass if it does exist,
    cancel the program if there is another error (e.g writing to protected directory)
    '''
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

import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PYTHON_INSTALL = 'python3'

# 获取当前文件的绝对路径和父目录
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# 将 script 目录的路径添加到 Python 解释器的搜索路径中
script_dir = os.path.join(os.path.dirname(current_dir), 'script')
sys.path.append(script_dir)

# 现在可以安全地导入 PATHS 变量
from paths import PATHS


# 获取目标文件夹名称
def get_target_name(path):
    """
    从路径中提取目标名称，假设目标名称为连续的四个大写字母或数字。
    """
    pattern = re.compile(r'[a-zA-Z0-9]{4,20}')
    match = pattern.search(path)
    return match.group(0) if match else 'Target'


# 检查目录是否存在文件
def directory_has_files(directory):
    return os.path.exists(directory) and any(os.scandir(directory))


# pdb文件清理步骤
def clean_pdb_files(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    clean_data_program = os.path.join(PATHS.sw_install, 'script/re_number_residue_index.pl')

    print(f"Cleaning PDBs from {input_directory} to {output_directory}...")

    for root, dirs, files in os.walk(input_directory):
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name)
            output_subdir = os.path.join(subdir_path, 'output')
            if os.path.exists(output_subdir):
                pdb_files = [f for f in os.listdir(output_subdir) if f.endswith('.pdb')]
                for pdb_file in pdb_files:
                    input_pdb_path = os.path.join(output_subdir, pdb_file)
                    output_pdb_path = os.path.join(output_directory, pdb_file)
                    command = f"perl {clean_data_program} {input_pdb_path} {output_pdb_path}"

                    try:
                        subprocess.run(command, shell=True, check=True)
                        print(f"Processed file: {input_pdb_path} -> {output_pdb_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing file {input_pdb_path}: {e}")

    print("Cleaned PDBs at:", output_directory)
    return output_directory


# step0: 为每个pdb文件添加链ID并组织到对应的target_name文件夹中
def add_chain_id(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    step0_location = os.path.join(PATHS.sw_install, 'script/step0_prepare_add_chain_to_folder.py')
    chain_add_location = os.path.join(PATHS.sw_install, 'script/assist_add_chainID_to_one_pdb.pl')

    # 调用 step0_prepare_add_chain_to_folder.py 脚本
    command = f'{PYTHON_INSTALL} {step0_location} {chain_add_location} {input_directory} {output_directory}'
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print('Step 0 done, files at:', output_directory)
    except subprocess.CalledProcessError as e:
        print(f"Error in step 0: {e}")

    # 获取输出目录中的所有PDB文件并将它们移动到相应的子文件夹中
    pdb_files = [f for f in os.listdir(output_directory) if f.endswith('.pdb')]
    for pdb_file in pdb_files:
        target_name = get_target_name(pdb_file)
        target_dir = os.path.join(output_directory, target_name)
        os.makedirs(target_dir, exist_ok=True)
        original_path = os.path.join(output_directory, pdb_file)
        new_path = os.path.join(target_dir, pdb_file)
        os.rename(original_path, new_path)
        print(f"Moved {pdb_file} to {target_dir}")

    print("Chain IDs added and files organized.")
    return output_directory


# 从PDB文件创建json文件
def create_json_from_pdb(input_directory, output_directory, stride_tool_path):
    os.makedirs(output_directory, exist_ok=True)
    step1_location = os.path.join(PATHS.sw_install, 'script/step1_create_json_from_PDB.py')
    json_command = f'{PYTHON_INSTALL} {step1_location} {stride_tool_path} {input_directory} {output_directory}'
    try:
        with open('./temp_out/stride_error.log', 'w') as error_log:
            subprocess.run(json_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=error_log)
        print('Step 1 done, JSON files at:', output_directory)
    except subprocess.CalledProcessError as e:
        print(f"Error in step 1: {e}")
        with open('./temp_out/stride_error.log', 'r') as error_log:
            print(error_log.read())

    return output_directory


if __name__ == "__main__":
    input_dir = r"../data/data/train/DG_tmp"
    temp_dir = r"temp_out/DG_temp/temp"
    json_dir = r"temp_out/DG_temp/json_output"
    stride_tool_path = r"../script/stride_bin/stride"

    # Step 0: Clean PDB files
    cleaned_pdb_dir = os.path.join(temp_dir, 'cleaned_pdbs')
    if not directory_has_files(cleaned_pdb_dir):
        cleaned_pdb_dir = clean_pdb_files(input_dir, cleaned_pdb_dir)
    else:
        print(f"Cleaned PDBs already exist at: {cleaned_pdb_dir}")

    # Step 1: Add chain ID and organize files
    step0_output_dir = os.path.join(temp_dir, 'step_0')
    if not directory_has_files(step0_output_dir):
        add_chain_id(cleaned_pdb_dir, step0_output_dir)
    else:
        print(f"Step 0 files already exist at: {step0_output_dir}")

    # Step 2: Create JSON files
    json_output_dir = json_dir
    if not directory_has_files(json_output_dir):
        create_json_from_pdb(step0_output_dir, json_output_dir, stride_tool_path)
    else:
        print(f"JSON files already exist at: {json_output_dir}")

    print("Processing complete. Final JSON files at:", json_output_dir)

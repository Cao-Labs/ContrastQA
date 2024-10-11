import os
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor


def process_file(json_file_path, output_file_path, step2_script_path, rfpredictions_locations, python_install):
    """处理单个JSON文件，调用 step2_generate_fragment_structures.py 脚本并生成PKL文件"""
    # Skip processing if the .pkl file already exists
    if os.path.exists(output_file_path):
        print(f'Skipping {json_file_path}, output file already exists at {output_file_path}')
        return

    # Execute step2_generate_fragment_structures.py script
    step2_command = [python_install, step2_script_path, json_file_path, rfpredictions_locations, output_file_path]
    error_log_path = output_file_path.replace(".pkl", ".log")
    try:
        with open(error_log_path, 'w') as error_log:
            subprocess.run(step2_command, check=True, stdout=subprocess.DEVNULL, stderr=error_log)
        print(f'Processed file {json_file_path} to {output_file_path}')
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {json_file_path}: {e}")
        with open(error_log_path, 'r') as error_log:
            print(error_log.read())


def create_pkl_from_json(input_directory, output_directory, step2_script_path, rfpredictions_locations, python_install):
    os.makedirs(output_directory, exist_ok=True)
    dir_pattern = re.compile(r'[a-zA-Z0-9]{4}')

    # 存储待处理的文件列表
    tasks = []

    for subdir in os.listdir(input_directory):
        subdir_path = os.path.join(input_directory, subdir)
        if os.path.isdir(subdir_path) and dir_pattern.match(subdir):
            output_subdir = os.path.join(output_directory, subdir)
            os.makedirs(output_subdir, exist_ok=True)

            for json_file in os.listdir(subdir_path):
                if json_file.endswith(".json"):
                    json_file_path = os.path.join(subdir_path, json_file)
                    output_file_path = os.path.join(output_subdir, json_file.replace(".json", ".pkl"))

                    # 将文件路径和处理函数参数加入任务列表
                    tasks.append(
                        (json_file_path, output_file_path, step2_script_path, rfpredictions_locations, python_install))

    # 并行处理任务
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.70)) as executor:
        futures = [executor.submit(process_file, *task) for task in tasks]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error in processing: {e}")


def directory_has_files(directory):
    has_files = any(os.scandir(directory))
    print(f"Directory {directory} has files: {has_files}")
    return has_files


if __name__ == "__main__":
    input_dir = r"temp_out/DG_temp/json_output"  # 输入目录
    output_dir = r"temp_out/DG_temp/pkl_output"  # 输出目录
    step2_script_path = r"../script/step2_generate_fragment_structures.py"  # step2_generate_fragment_structures.py 脚本路径
    rfpredictions_locations = r'../script/assist_generation_scripts/RF_Predictions/'  # 随机森林预测路径
    python_install = "python3"  # Python 解释器路径

    # Step 2: Create PKL files from JSON files
    create_pkl_from_json(input_dir, output_dir, step2_script_path, rfpredictions_locations, python_install)

    print("Final PKL files at:", output_dir)

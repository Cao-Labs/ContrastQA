import os
import torch
import pickle


def merge_pt_files_to_pkl(subdir, output_dir):
    """
    Merge .pt files in a subdirectory and save the result as a .pkl file.

    Parameters:
    - subdir: str, path to the subdirectory containing .pt files.
    - output_dir: str, path to the directory where the merged .pkl file will be saved.
    """
    subdir_name = os.path.basename(subdir).replace("_emb", "")

    # 列出子文件夹中所有的 .pt 文件
    pt_files = [f for f in os.listdir(subdir) if f.endswith('.pt')]
    merged_data = {}

    # 遍历每个 .pt 文件并加载其内容
    for pt_file in pt_files:
        pt_path = os.path.join(subdir, pt_file)
        data = torch.load(pt_path)

        # 如果数据是列表，遍历列表并提取每个字典的 'label' 和 'representations'
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    chain_name = item['label']
                    representations = item['representations']
                    # 如果 representations 是 Tensor，直接保存
                    if isinstance(representations, torch.Tensor):
                        merged_data[chain_name] = representations.numpy()
                    # 如果 representations 是字典，按原方式处理
                    elif isinstance(representations, dict):
                        merged_data[chain_name] = {layer: rep.numpy() for layer, rep in representations.items()}
                    else:
                        print(f"Skipping representations in {pt_file} because it's neither Tensor nor dict.")
                else:
                    print(f"Skipping item: {item} because it's not a dictionary.")
        elif isinstance(data, dict):
            # 如果数据是字典，按照原方法提取
            chain_name = data['label']
            representations = data['representations']
            # 如果 representations 是 Tensor，直接保存
            if isinstance(representations, torch.Tensor):
                merged_data[chain_name] = representations.numpy()
            # 如果 representations 是字典，按原方式处理
            elif isinstance(representations, dict):
                merged_data[chain_name] = {layer: rep.numpy() for layer, rep in representations.items()}
            else:
                print(f"Skipping representations in {pt_file} because it's neither Tensor nor dict.")
        else:
            print(f"Skipping {pt_file} because its data is neither a dictionary nor a list.")

    # 将合并后的数据保存为 .pkl 文件
    output_file = os.path.join(output_dir, f"{subdir_name}_emb.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"Saved merged embeddings for {subdir_name} to {output_file}")


def save_esm_emb_pkl(input_dir, output_dir):
    """
    Process all subdirectories with a "_emb" suffix and save merged .pkl files for each.

    Parameters:
    - input_dir: str, directory containing subdirectories with .pt files.
    - output_dir: str, directory where the merged .pkl files will be saved.
    """
    # 遍历输入目录下的所有子文件夹
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # 只处理后缀为 "_emb" 的子文件夹
        if os.path.isdir(subdir_path) and subdir.endswith("_emb"):
            # 直接将 .pkl 文件保存到输出目录，不再创建子文件夹
            merge_pt_files_to_pkl(subdir_path, output_dir)


# 如果这个文件是作为脚本运行的，则执行以下代码
if __name__ == "__main__":
    input_directory = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\MUL_embedding"
    output_directory = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\MUL_embedding_pkl"

    # 只有当直接运行该脚本时才会执行这段代码
    save_esm_emb_pkl(input_directory, output_directory)

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

    pt_files = [f for f in os.listdir(subdir) if f.endswith('.pt')]
    merged_data = {}

    for pt_file in pt_files:
        pt_path = os.path.join(subdir, pt_file)
        data = torch.load(pt_path)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    chain_name = item['label']
                    representations = item['representations']
                    if isinstance(representations, torch.Tensor):
                        merged_data[chain_name] = representations.numpy()
                    elif isinstance(representations, dict):
                        merged_data[chain_name] = {layer: rep.numpy() for layer, rep in representations.items()}
                    else:
                        print(f"Skipping representations in {pt_file} because it's neither Tensor nor dict.")
                else:
                    print(f"Skipping item: {item} because it's not a dictionary.")
        elif isinstance(data, dict):
            chain_name = data['label']
            representations = data['representations']
            if isinstance(representations, torch.Tensor):
                merged_data[chain_name] = representations.numpy()
            elif isinstance(representations, dict):
                merged_data[chain_name] = {layer: rep.numpy() for layer, rep in representations.items()}
            else:
                print(f"Skipping representations in {pt_file} because it's neither Tensor nor dict.")
        else:
            print(f"Skipping {pt_file} because its data is neither a dictionary nor a list.")

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
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if os.path.isdir(subdir_path) and subdir.endswith("_emb"):
            merge_pt_files_to_pkl(subdir_path, output_dir)

# example usages
if __name__ == "__main__":
    input_directory = r"data\temp_out\MUL_embedding"
    output_directory = r"data\temp_out\MUL_embedding_pkl"

    save_esm_emb_pkl(input_directory, output_directory)

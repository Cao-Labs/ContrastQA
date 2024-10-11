import os
import pickle
import subprocess
import re
import warnings

from model.dataload import ProteinGVPQAData_Generator, load_esm_embeddings
from script.paths import PATHS

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PYTHON_INSTALL = 'python3'

# 设置数据路径和其他参数
processed_data_path = '../data/temp_out/MUL/pkl_output'
esm_data_path = '../data/temp_out/MUL_embedding_pkl/7N1J_embedding'
output_data_path = '../data/temp_out/MUL_DGL_tmp'

# 创建数据生成器实例
data_gen = ProteinGVPQAData_Generator(
    data_path=processed_data_path,
    esm_path=esm_data_path,
    protein_info_pickle='../data/temp_out/protein_info/MUL/7N1J_info.csv',
    min_seq_size=0,
    max_seq_size=10000,
    batch_size=1,
    max_msa_seq=100000,
    max_id_nums=1000000
)

# 获取数据集
dataset = data_gen.get_dataset()

# 确保输出目录存在
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)


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
    print(f"Processing {pkl_filename} and {esm_filename}:")

    # 转换为 DGL 图
    dgl_graphs = data_gen.convert_to_dgl(node_features.numpy(), contact_map.numpy())
    dgl_filename = pkl_filename.replace('.pkl', '.dgl')
    filepath = os.path.join(output_data_path, dgl_filename)

    if dgl_graphs:
        data_gen.save_dgl_graphs(dgl_graphs, filepath)
        print(f"Processed {pkl_filename} and {esm_filename} -> {dgl_filename}")
    else:
        print(f"Warning: No graphs generated for {pkl_filename}")

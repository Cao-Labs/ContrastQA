import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import Selection

warnings.filterwarnings("ignore")

pdb_parser = PDB.PDBParser()


# 计算两个氨基酸之间欧几里得距离的函数
# Calculate the Euclidean distance of two amino acids
# 首先计算每个坐标维度之间的差值，然后将这些差值的平方相加，最后取平方根得到距离
def calculate_distance(residue_one, residue_two):
    sum = 0
    distance_list = [residue_one[i] - residue_two[i] for i in range(len(residue_one))]
    for i in range(len(distance_list)):
        sum += math.pow(distance_list[i], 2)
    distance = np.sqrt(sum)
    return distance


# 获取一个蛋白质结构中的所有链对象，并返回它们的列表
def get_num_chains(structure):
    '''
    Returns a list of chain objects in a structure
    '''
    ch_list = Selection.unfold_entities(structure, 'C')
    return ch_list


# 计算给定PDB文件中的链（chains）数量
def calculate_num_chains(pdbfile,
                         out=None):
    structure = pdb_parser.get_structure("id", pdbfile)
    ch_list = get_num_chains(structure)

    return ch_list

# # 使用示例
# pdbfile_path = r"D:\pycharm\pycharmProjects\ideamodel2qa\ideamodel2qa\data\data\1JTD\1JTD_correct.pdb"
# atom_index_list, atom_type_list, coordinates_list = extract_atoms_with_types(pdbfile_path)
# one_hot_encoded_types = one_hot_encode_atom_types(atom_type_list)
# combined_features = combine_features(atom_index_list, one_hot_encoded_types, coordinates_list)
#
# print(combined_features)


# 用于解析一个PDB文件并提取每个链中所有Cα（alpha carbon）原子的坐标和相关信息
def calculate_interface(pdbfile):
    count = 0
    chain_number = 0
    index = 0
    atom_index = 0
    chain_id = '#'
    count_list = []
    index_list = []
    coordinate_list = []
    chain_number_list = []

    with open(pdbfile) as pdb:
        for line in pdb:
            # print((lne[13:16]))
            # 检查当前行是否为Cα原子的行
            if (line[13:16] == 'CA '):
                # 检查是否为同一条链且当前索引大于上一个索引
                if (int(line[22:26]) > index and line[21] == chain_id):
                    # 更新原子索引、计数、索引列表、坐标列表和链编号列表
                    atom_index = int(line[6:11])
                    count = count + 1
                    count_list.append(count)
                    index = int(line[22:26])  # pdb氨基酸的序号
                    index_list.append(index)
                    coordinate_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    chain_number_list.append(chain_number)
                # 检查是否为新链或非连续原子
                elif (int(line[22:26]) < index or line[21] != chain_id):
                    atom_index = int(line[6:11])
                    chain_id = line[21]
                    chain_number = chain_number + 1
                    count = count + 1
                    count_list.append(count)
                    index = int(line[22:26])
                    index_list.append(index)
                    coordinate_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    chain_number_list.append(chain_number)
            else:
                continue
        # 返回包含索引、坐标、计数和链编号的四个列表
        return index_list, coordinate_list, count_list, chain_number_list


# 用于计算PDB文件中各链之间的界面残基索引
def calculate_interface_index(index_list, coordinate, count_list, chain_list):
    # 用于存储所有链和其残基信息的字典
    DB = dict()
    for i in range(chain_list[-1]):
        chain = 'chain' + str(i + 1)
        DB[chain] = dict()
    # print(DB)
    # 遍历chain_list，构建每个链及其残基的信息，并存储在DB字典中
    for i in range(len(chain_list)):
        chain = 'chain' + str(chain_list[i])
        DB[chain][count_list[i]] = dict()
        DB[chain][count_list[i]]['index'] = index_list[i]
        DB[chain][count_list[i]]['coordinate'] = coordinate[i]

    # 提取DB中的所有链，并组织成列表chain_list_all，每个元素包含一个链的所有残基信息（计数和坐标）
    key_list = list(DB.keys())
    # print(DB)
    chain_list_all = [[] for i in range(len(key_list))]
    for i in range(len(key_list)):
        key_list2 = list(DB[key_list[i]])
        for j in range(len(key_list2)):
            count = key_list2[j]
            # print(count)
            x = DB[key_list[i]][key_list2[j]]['coordinate'][0]
            y = DB[key_list[i]][key_list2[j]]['coordinate'][1]
            z = DB[key_list[i]][key_list2[j]]['coordinate'][2]
            chain_list_all[i].append([count, x, y, z])

    interface_index = set()
    interface_mask = [0 for i in range(len(count_list))]

    # 遍历所有链的所有残基，计算链间残基距离
    for i in range(len(chain_list_all)):
        for j in range(i + 1, len(chain_list_all)):
            for k in range(len(chain_list_all[i])):
                for f in range(len(chain_list_all[j])):
                    residue_one = chain_list_all[i][k][1:]
                    residue_two = chain_list_all[j][f][1:]
                    one_index = chain_list_all[i][k][0]
                    two_index = chain_list_all[j][f][0]
                    distance = calculate_distance(residue_one, residue_two)
                    # 如果距离小于8Å，则认为是界面残基，将其索引添加到interface_index集合中
                    if (distance < 8):
                        interface_index.add(one_index)
                        interface_index.add(two_index)

    # 将界面残基索引排序并生成interface_mask，其中界面残基的索引对应的位置设为1
    index_sort = sorted(list(interface_index))
    for i in range(len(index_sort)):
        index = index_sort[i]
        interface_mask[index - 1] = 1

    # 返回排序后的界面残基索引和掩码
    return index_sort, interface_mask


# 用于将PDB文件中的复合物转换为距离矩阵表示
def complex2map(pdbfile):
    index_list, coordinate, count_list, chain_list = calculate_interface(pdbfile)

    # 初始化字典DB用于存储每条链的残基信息
    DB = dict()  # Store protein chains, location and index information
    for i in range(chain_list[-1]):
        chain = 'chain' + str(i + 1)
        DB[chain] = dict()
    # print(DB)
    # 遍历chain_list，构建每个链及其残基的信息，并存储在DB字典中
    for i in range(len(chain_list)):
        chain = 'chain' + str(chain_list[i])
        DB[chain][count_list[i]] = dict()
        DB[chain][count_list[i]]['index'] = index_list[i]
        DB[chain][count_list[i]]['coordinate'] = coordinate[i]

    # 提取DB中的所有链，并组织成列表chain_list_all，每个元素包含一个链的所有残基信息（计数和坐标）
    key_list = list(DB.keys())
    # pprint(DB)
    chain_list_all = [[] for i in range(len(key_list))]
    for i in range(len(key_list)):
        key_list2 = list(DB[key_list[i]])
        for j in range(len(key_list2)):
            count = key_list2[j]
            # print(count)
            x = DB[key_list[i]][key_list2[j]]['coordinate'][0]
            y = DB[key_list[i]][key_list2[j]]['coordinate'][1]
            z = DB[key_list[i]][key_list2[j]]['coordinate'][2]
            chain_list_all[i].append([count, x, y, z])

    # 初始化chain2map字典，用于存储每个链之间的距离矩阵
    chain2map = {}
    # 遍历所有链的组合，计算每对残基之间的距离
    for i in range(len(chain_list_all)):
        chain2map[i] = []
        for j in range(0, len(chain_list_all)):
            if i == j:
                continue
            # 使用np.full初始化一个全为NaN的矩阵compelx_map，然后填充距离值
            compelx_map = np.full((len(chain_list_all[i]), len(chain_list_all[j])), np.nan)
            for k in range(len(chain_list_all[i])):
                for f in range(len(chain_list_all[j])):
                    residue_one = chain_list_all[i][k][1:]
                    residue_two = chain_list_all[j][f][1:]
                    one_index = chain_list_all[i][k][0]
                    two_index = chain_list_all[j][f][0]
                    distance = calculate_distance(residue_one, residue_two)
                    compelx_map[k][f] = distance
            chain2map[i].append(compelx_map)

    # 返回包含各链之间距离矩阵的字典chain2map
    return chain2map


# 用于生成界面残基的信息
def generate_interface_info(pdbfile):
    # pdbfile: 输入的PDB文件路径。
    # 调用 calculate_interface函数，获取残基索引、坐标、计数和链编号。
    # 调用 calculate_interface_index函数，获取界面残基索引和掩码。
    # 初始化 interface_res_info 列表，用于存储界面残基信息
    index_list, coordinate, count_list, chain_list = calculate_interface(pdbfile)
    interface_index, interface_mask = calculate_interface_index(index_list, coordinate, count_list, chain_list)
    interface_res_info = []
    if len(interface_index) == 0:
        return interface_res_info, interface_mask
    else:
        for i in range(len(interface_index)):
            # 获取链名称列表
            chains_name_list = calculate_num_chains(pdbfile)
            count_index = interface_index[i] - 1
            chain_index = chain_list[count_index] - 1
            # 确定残基所在的链，并获取链的名称和残基的编号
            chain_name = chains_name_list[chain_index].get_id()
            index = index_list[count_index]
            # 将链名称和残基编号格式化为字符串并添加到 interface_res_info 列表中
            interface_res_info.append(f'{chain_name}:{index}')
    return interface_res_info, interface_mask


# 用于计算蛋白质复合物中每个链的界面残基掩码
def get_complex_interface(pred_chain2map, T=8):
    import pandas as pd

    # return a 1D boolean array indicating where the distance in the
    # upper triangle meets the threshold comparison
    # 用于返回一个布尔数组，指示距离矩阵中的距离是否满足阈值比较
    def get_dist_thresh_b_indices(dmap_flat, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        if comparator == 'gt':
            threshed = dmap_flat > thresh
        elif comparator == 'lt':
            threshed = dmap_flat < thresh
        elif comparator == 'ge':
            threshed = dmap_flat >= thresh
        elif comparator == 'le':
            threshed = dmap_flat <= thresh
        return threshed

    # 用于返回一个布尔数组，指示距离矩阵中的距离是否满足阈值比较
    interface_mask = []
    chain_list = []
    # 遍历每个链的预测距离矩阵列表，并将其连接成一个大的矩阵
    for chainid in pred_chain2map.keys():
        pred_map_list = pred_chain2map[chainid]

        pred_map = np.concatenate(pred_map_list, axis=1)

        # 遍历每一行，计算界面残基对的索引
        for i in range(len(pred_map)):
            chain_list.append(chainid)

            pred_flat_map = pred_map[i]

            interface_thresh_indices = get_dist_thresh_b_indices(pred_flat_map, T, 'lt')

            interface_n = interface_thresh_indices.sum()
            # print(interface_n)

            # 通过阈值比较确定界面残基
            if interface_n > 0:
                interface_mask.append(1)
            else:
                interface_mask.append(0)

    return interface_mask


# We have improved the lddt metric for calculating protein complexes
# 用于计算蛋白质复合物的局部距离差异测试（lDDT）得分。此函数比较参考结构和预测结构之间的距离差异，以评估预测模型的准确性

def get_complex_LDDT(ref_chain2map, pred_chain2map, R=15, T=8, sep_thresh=-1, T_set=[0.5, 1, 2, 4], precision=4):
    '''
    Mariani V, Biasini M, Barbato A, Schwede T.
    lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests.
    Bioinformatics. 2013 Nov 1;29(21):2722-8.
    doi: 10.1093/bioinformatics/btt473.
    Epub 2013 Aug 27.
    PMID: 23986568; PMCID: PMC3799472.
    '''
    # ref_chain2map: 参考结构的链到距离矩阵的映射。
    # pred_chain2map: 预测结构的链到距离矩阵的映射。
    # R: 定义界面残基对的距离阈值。
    # T: 界面残基对之间距离的阈值，用于界定是否属于界面。
    # sep_thresh: 不在这个函数中使用。
    # T_set: 计算lDDT时使用的阈值集合。
    # precision: lDDT得分的小数点精度。
    import pandas as pd

    # return a 1D boolean array indicating where the distance in the
    # upper triangle meets the threshold comparison
    # 返回一个布尔数组，指示上三角矩阵中的距离是否满足阈值比较
    def get_dist_thresh_b_indices(dmap_flat, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        if comparator == 'gt':
            threshed = dmap_flat > thresh
        elif comparator == 'lt':
            threshed = dmap_flat < thresh
        elif comparator == 'ge':
            threshed = dmap_flat >= thresh
        elif comparator == 'le':
            threshed = dmap_flat <= thresh
        return threshed

    # Helper for number preserved in a threshold
    # 返回在给定阈值下保留的元素数量
    def get_n_preserved(ref_flat, mod_flat, thresh):
        err = np.abs(ref_flat - mod_flat)
        n_preserved = (err < thresh).sum()
        return n_preserved

    # 初始化存储lDDT得分、界面掩码和链列表的变量
    lddt_scores = []
    interface_mask = []
    chain_list = []
    # 遍历每个链，获取预测和参考的距离矩阵列表，并将其连接成一个大的矩阵
    for chainid in pred_chain2map.keys():
        pred_map_list = pred_chain2map[chainid]
        ref_map_list = ref_chain2map[chainid]

        true_map = np.concatenate(ref_map_list, axis=1)

        pred_map = np.concatenate(pred_map_list, axis=1)

        for i in range(len(true_map)):
            chain_list.append(chainid)
            true_flat_map = true_map[i]
            pred_flat_map = pred_map[i]

            # 遍历每一行，计算界面残基对的索引
            # Find set L
            R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, R, 'lt')
            # print(R_thresh_indices)
            interface_thresh_indices = get_dist_thresh_b_indices(pred_flat_map, T, 'lt')
            # print(interface_thresh_indices)
            L_indices = R_thresh_indices

            true_flat_in_L = true_flat_map[L_indices]
            # print(true_flat_in_L)
            pred_flat_in_L = pred_flat_map[L_indices]
            # print(pred_flat_in_L)

            # Number of pairs in L
            L_n = L_indices.sum()
            # print(L_n)
            interface_n = interface_thresh_indices.sum()

            # 通过阈值比较确定界面残基
            if interface_n > 0:
                interface_mask.append(1)
            else:
                interface_mask.append(0)

            # 计算并存储保留的分数，计算并存储lDDT得分
            # Calculated lDDT
            preserved_fractions = []
            for _thresh in T_set:
                _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, _thresh)

                if L_n == 0:
                    _f_preserved = 0
                else:
                    _f_preserved = _n_preserved / L_n
                preserved_fractions.append(_f_preserved)

            # preserved_fractions

            lDDT = np.mean(preserved_fractions)
            if precision > 0:
                lDDT = round(lDDT, precision)

            # print(i,": ",preserved_fractions, lDDT)
            lddt_scores.append(lDDT)

    return lddt_scores


# 用于从指定的输入目录中读取PDB文件，生成界面信息，并将结果保存到CSV文件中
# inputfile: 输入目录，包含多个目标，每个目标包含多个PDB文件。
# outfile: 输出的CSV文件路径
def prepare_data_information(inputfile, outfile):
    # 初始化四个列表，分别用于存储目标名称、复合物名称、界面掩码和界面残基信息
    target_list = []
    complex_all = []
    mask = []
    interface_residue = []

    # 外层循环遍历每个目标目录
    for target in os.listdir(inputfile):
        target_path = os.path.join(inputfile, target)
        output_path = os.path.join(target_path, 'output')

        # 检查是否存在 output 文件夹
        if os.path.isdir(output_path):
            # 内层循环遍历 output 文件夹中的PDB文件
            for pdb in os.listdir(output_path):
                try:
                    complex_name = pdb
                    complex_all.append(complex_name)
                    # 获取PDB文件的完整路径并存储相关信息
                    file_path = os.path.join(output_path, pdb)
                    target_list.append(target)
                    # pred_chain2map = complex2map(file_path)
                    # 调用 generate_interface_info 函数生成界面残基信息和界面掩码
                    interface_res_info, interface_mask = generate_interface_info(file_path)

                    # interface_mask = get_complex_interface(pred_chain2map)

                    mask.append(interface_mask)
                    interface_residue.append(interface_res_info)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    # 将收集到的信息存储在一个Pandas数据框中
    dataframe = pd.DataFrame(
        {'target': target_list,
         'model': complex_all,
         'interface_residue': interface_residue,
         'interface_mask': mask})
    # 将数据框保存为CSV文件
    dataframe.to_csv(outfile, index=False, sep=',')


# 用于从指定的输入目录中读取PDB文件，与本地(native)文件进行比较，计算LDDT分数，并将结果保存到CSV文件中
# inputfile: 输入目录，包含多个目标，每个目标包含多个PDB文件。
# native_file: 本地PDB文件路径，用于参考比较。
# outfile: 输出的CSV文件路径
def prepare_information_with_native(inputfile, native_file, outfile):
    # 初始化五个列表，分别用于存储目标名称、复合物名称、LDDT分数、界面掩码和界面残基信息。
    # 通过 complex2map 函数计算本地文件的链到距离矩阵的映射 ref_chain2map
    target_list = []
    complex_all = []
    scores = []
    mask = []
    interface_residue = []
    ref_chain2map = complex2map(native_file)
    # 外层循环遍历每个目标目录
    for target in os.listdir(inputfile):
        target_path = os.path.join(inputfile, target)
        # 内层循环遍历每个目标目录中的PDB文件
        for pdb in os.listdir(target_path):
            complex_name = pdb
            complex_all.append(complex_name)
            # 获取PDB文件的完整路径并存储相关信息
            file_path = os.path.join(inputfile, target, pdb)
            target_list.append(target)
            # 通过 complex2map 计算预测文件的链到距离矩阵的映射 pred_chain2map
            pred_chain2map = complex2map(file_path)
            # 通过 get_complex_LDDT 计算预测结构与参考结构之间的LDDT分数
            lddt_complex = get_complex_LDDT(ref_chain2map, pred_chain2map, R=30, T=8, sep_thresh=-1,
                                            T_set=[0.5, 1, 2, 4], precision=4)
            # 调用 generate_interface_info 生成界面残基信息和界面掩码
            interface_res_info, interface_mask = generate_interface_info(file_path)
            scores.append(lddt_complex)
            interface_residue.append(interface_res_info)
            mask.append(interface_mask)
    # 生成数据框并保存为CSV文件
    dataframe = pd.DataFrame(
        {'target': target_list,
         'model': complex_all,
         'lddt_complex': scores,
         'interface_residue': interface_residue,
         'interface_mask': mask})
    dataframe.to_csv(outfile, index=False, sep=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare data information')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="number of pdbs to use for each batch")
    parser.add_argument('-n', '--native', help='native PDB file ', default=None)
    parser.add_argument('-o', '--outfile', type=str, required=True, help="saving complex information")

    args = parser.parse_args()

    inputfile = args.input_dir
    native_pdb = args.native
    outfile = args.outfile

    if args.native != None:
        prepare_information_with_native(inputfile, native_pdb, outfile)

    else:
        prepare_data_information(inputfile, outfile)


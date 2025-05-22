# ContrastQA: A label-guided graph contrastive learning-based approach for protein complex structure quality assessment

<div align="center">
  <img src="single_overview.pdf" width="100%"/>
</div>

We propose the ContrastQA method, based on the label-guided graph contrastive learning framework. ContrastQA devises a novel positive-negative samples pair selection strategy utilizing the interface score label DockQ and also gives a modified contrast loss function to fit this task, which is combined with the geometric graph neural network GVP-GNN for contrastive learning to enhance the model representation and thus improve the global quality assessment of protein complexes. 

## Installation

```bash
# clone the repo
git clone https://github.com/Cao-Labs/ContrastQA.git
cd ContrastQA
```
We use python 3.9.19, pytorch 2.1.2 and pytorch-lightning 2.3.0. We recommend using conda to install the dependencies:
```bash
conda env create -f environment.yml
```
Activate conda environment
```bash
conda activate ContrastQA
```
You also need to install the relative packages to run ESM-2 protein language model. \
Please see [facebookresearch/esm](https://github.com/facebookresearch/esm) for details. 

## Dataset
### Training sets
We provide datasets for training, including MULTICOM, Voro, and PPI datasets(generated using AF3 and AF2-multimer).

### Testing sets
We use CASP16, which is available from the CASP official website, and ABAG-AF3, which is available from TopoQA, as our testing datasets.

## Usage
### Model Training:
We start by generating the dgl file to train the model：
```bash
python ./data/data_generator.py
--input_pdb_folder -i input pdbs folder
--fasta_folder -f input fasta folder
--dgl_save_folder -o dgl files save folder
--esm_pkl_folder -e input esm pkl folder
--cores  -c number of cores for parallel processing

# example code
python ./src/data_generator.py -i /example_pdbs_folder/ -f /example_fasta_folder/ -o /dgl_save_folder -e /example_esm_pkl_folder/ -c 10
```
### Model Test:
ContrastQA requires GPU. We provide few protein complex pdb files for testing, you can use the provided model weight to predict protein complex structures' quality. The evaluation result Rank_[targetid]_qa.csv is stored in result_folder. We benchmark our model by running inference.py:
```bash
python ./inference.py ./example/7sgm/ ./example/result/7sgm/
```
You are free to evaluate your own dataset, which is in the following format:
```bash
data_folder
├── decoy_1.pdb
├── decoy_2.pdb
├── decoy_3.pdb
├── decoy_4.pdb
└── decoy_5.pdb
```

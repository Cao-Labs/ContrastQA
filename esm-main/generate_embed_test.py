import os
import subprocess
from Bio import PDB
from Bio.SeqUtils import seq1


def pdb_to_fasta(pdb_file, fasta_file):
    """
    Convert a PDB file to a FASTA file.

    Parameters:
    - pdb_file: str, path to the PDB file.
    - fasta_file: str, path to the output FASTA file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)

    seqs = []
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    seq.append(seq1(residue.resname))
            seqs.append(f">chain_{chain.id}\n{''.join(seq)}")

    with open(fasta_file, "w") as f:
        f.write("\n".join(seqs))


def extract_embeddings(model_location, fasta_dir, output_dir, toks_per_batch=4096, repr_layers=[33],
                       include=["per_tok"], truncation_seq_length=1022):
    """
    Extract embeddings from all FASTA files in a directory using the ESM model.

    Parameters:
    - model_location: str, path to the pretrained model or the model name.
    - fasta_dir: str, directory containing FASTA files.
    - output_dir: str, directory to save the extracted embeddings.
    - toks_per_batch: int, maximum batch size for token processing.
    - repr_layers: list of int, layer indices from which to extract representations.
    - include: list of str, specify which representations to return.
    - truncation_seq_length: int, truncate sequences longer than the given value.
    """

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all FASTA files in the input directory
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]

    extract_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'monomer_seq_emb.py'))

    for fasta_file in fasta_files:
        fasta_path = os.path.join(fasta_dir, fasta_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(fasta_file)[0]}_emb")

        command = [
            "python", extract_script,
            model_location,
            fasta_path,
            output_path,
            "--toks_per_batch", str(toks_per_batch),
            "--repr_layers", *map(str, repr_layers),
            "--include", *include,
            "--truncation_seq_length", str(truncation_seq_length)
        ]

        # Run the extraction command
        print(f"Extracting embeddings for {fasta_file}...")
        subprocess.run(command, check=True)
        print(f"Embeddings for {fasta_file} saved to {output_path}")


def process_pdb_files_in_dir(pdb_dir, fasta_dir, model_location, output_base_dir):
    """
    Process PDB files in a directory: convert to FASTA and extract embeddings.

    Parameters:
    - pdb_dir: str, the directory containing PDB files (can be a single folder or subfolders).
    - fasta_dir: str, directory where the FASTA files will be stored.
    - model_location: str, path to the pretrained model or the model name.
    - output_base_dir: str, base directory to save the extracted embeddings.
    """
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    # If the input directory contains files directly (no subfolders), process them
    if not any(os.path.isdir(os.path.join(pdb_dir, subfolder)) for subfolder in os.listdir(pdb_dir)):
        print(f"Processing PDB files in directory: {pdb_dir}")
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        for pdb_file in pdb_files:
            pdb_path = os.path.join(pdb_dir, pdb_file)
            fasta_file = os.path.join(fasta_dir, f"{os.path.splitext(pdb_file)[0]}.fasta")
            pdb_to_fasta(pdb_path, fasta_file)

        # Extract embeddings from the FASTA files
        extract_embeddings(model_location, fasta_dir, output_base_dir)

    # If there are subdirectories (e.g., 1ay7, 1buh), process each subfolder separately
    else:
        for subfolder in os.listdir(pdb_dir):
            subfolder_path = os.path.join(pdb_dir, subfolder)

            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder}")
                subfolder_fasta_dir = os.path.join(fasta_dir, subfolder)
                if not os.path.exists(subfolder_fasta_dir):
                    os.makedirs(subfolder_fasta_dir)

                # Process PDB files in this subfolder
                pdb_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pdb')]
                for pdb_file in pdb_files:
                    pdb_path = os.path.join(subfolder_path, pdb_file)
                    fasta_file = os.path.join(subfolder_fasta_dir, f"{os.path.splitext(pdb_file)[0]}.fasta")
                    pdb_to_fasta(pdb_path, fasta_file)

                # Extract embeddings for this subfolder
                subfolder_output_dir = os.path.join(output_base_dir, subfolder)
                extract_embeddings(model_location, subfolder_fasta_dir, subfolder_output_dir)


def generate_embed_pt(input_dir, model_location, output_base_dir):
    """
    Main function to handle both single directory and multiple subfolders.

    Parameters:
    - input_dir: str, the directory containing PDB files (single folder or subfolders).
    - model_location: str, path to the pretrained model or the model name.
    - output_base_dir: str, base directory to save the extracted embeddings.
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Create the FASTA directory inside the output base directory
    fasta_dir = os.path.join(output_base_dir, "fasta")
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    # Process the PDB files (either in the main directory or in subfolders)
    process_pdb_files_in_dir(input_dir, fasta_dir, model_location, output_base_dir)


if __name__ == "__main__":
    # Example usage
    input_dir = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\data\test\CASP15\test_tmp\H1157_split\chain_C"  # Input directory (can be a single folder or subfolders)
    model_location = "esm2_t33_650M_UR50D"  # Path to your pretrained model
    output_base_dir = r"D:\pycharm\pycharmProjects\ideamodel3qa\data\temp_out\CASP15_embedding\H1157_C"  # Output directory to save embeddings

    generate_embed_pt(input_dir, model_location, output_base_dir)

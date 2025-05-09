import torch
import os
import numpy as np
from Bio import pairwise2


def get_dssp(fasta_file, pdb_path):
    DSSP = '/home/pplion/anaconda3/envs/myqa/bin/dssp'

    def process_dssp(dssp_file):
        print(f"Reading DSSP file: {dssp_file}")
        aa_type = "ACDEFGHIKLMNPQRSTVWY"  # Define standard amino acids
        SS_type = "HBEGITSC"  # Define secondary structure types
        rASA_std = [
            115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
            185, 160, 145, 180, 225, 115, 140, 155, 255, 230,
        ]  # Standard relative accessible surface areas

        with open(dssp_file, "r") as f:  # Open DSSP output file
            lines = f.readlines()  # Read all lines

        seq = ""  # Initialize sequence string
        dssp_feature = []  # Initialize list to store DSSP features

        p = 0
        while lines[p].strip()[0] != "#":  # Find the line starting with DSSP data
            p += 1
        for i in range(p + 1, len(lines)):  # Iterate over DSSP data lines
            aa = lines[i][13]  # Extract amino acid type
            if aa == "!" or aa == "*":  # Skip unknown residues
                continue
            seq += aa  # Append amino acid to sequence
            SS = lines[i][16]  # Extract secondary structure type
            if SS == " ":
                SS = "C"  # Default to coil if no secondary structure is specified
            SS_vec = np.zeros(9)  # Initialize a vector for secondary structure one-hot encoding
            SS_vec[SS_type.find(SS)] = 1  # Set the appropriate position to 1
            PHI = float(lines[i][103:109].strip())  # Extract phi angle
            PSI = float(lines[i][109:115].strip())  # Extract psi angle
            ACC = float(lines[i][34:38].strip())  # Extract accessible surface area
            ASA = min(100, round(ACC / rASA_std[aa_type.find(aa)] * 100)) / 100  # Normalize ASA value
            dssp_feature.append(np.concatenate((np.array([PHI, PSI, ASA]), SS_vec)))  # Append DSSP features

        return seq, dssp_feature  # Return sequence and DSSP features

    def match_dssp(seq, dssp, ref_seq):
        alignments = pairwise2.align.globalxx(ref_seq, seq)  
        ref_seq = alignments[0].seqA  
        seq = alignments[0].seqB  

        SS_vec = np.zeros(9)  
        SS_vec[-1] = 1
        padded_item = np.concatenate((np.array([360, 360, 0]), SS_vec))  

        new_dssp = [] 
        for aa in seq:
            if aa == "-": 
                new_dssp.append(padded_item) 
            else:
                new_dssp.append(dssp.pop(0))

        matched_dssp = []  
        for i in range(len(ref_seq)):
            if ref_seq[i] == "-": 
                continue
            matched_dssp.append(new_dssp[i])

        return new_dssp 

    def transform_dssp(dssp_feature):
        dssp_feature = np.array(dssp_feature)  # Convert DSSP features to NumPy array
        angle = dssp_feature[:, 0:2]  # Extract phi and psi angles
        ASA_SS = dssp_feature[:, 2:]  # Extract ASA and secondary structure vectors
        radian = angle * (np.pi / 180)  # Convert angles from degrees to radians
        dssp_feature = np.concatenate([np.sin(radian), np.cos(radian), ASA_SS],
                                      axis=1)  # Transform angles to sin and cos

        return dssp_feature  # Return transformed DSSP features

    def get_dssp_pdb(pdb_path, ref_seq):
        try:
            dssp_file = pdb_path + ".dssp"
            os.system(f"{DSSP} -i {pdb_path} -o {dssp_file}")  # Run DSSP to generate DSSP file

            if not os.path.exists(dssp_file):  
                print(f"DSSP file not generated for {pdb_path}, skipping...")
                return None

            dssp_seq, dssp_matrix = process_dssp(dssp_file)  # Process the DSSP output
            if dssp_seq != ref_seq:  # If sequences do not match
                dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)  # Align and match DSSP features

            dssp_tensor = torch.tensor(transform_dssp(dssp_matrix), dtype=torch.float32)  # Convert to tensor

            os.remove(dssp_file)  # Remove the DSSP file after processing
            return dssp_tensor  # Return the DSSP features as tensor
        except Exception as e:  # Catch exceptions
            print(f"Error processing {pdb_path}: {e}")  # Print exception message
            return None

    # Read the FASTA file and extract sequences
    pdbfasta = {}  # Dictionary to map PDB file names to sequences
    with open(fasta_file) as r1:
        fasta_ori = r1.readlines()  # Read all lines

    for i in range(len(fasta_ori)):
        if fasta_ori[i][0] == ">":  # Header line in FASTA
            name = fasta_ori[i].split(">")[1].replace("\n", "")
            seq = fasta_ori[i + 1].replace("\n", "")
            pdbfasta[name] = seq  # Map name to sequence
            print(name)
    fault_name = []  # List to store names of sequences that failed DSSP processing
    dssp_tensors = []  # List to store DSSP tensors

    for name in pdbfasta.keys():  # Iterate through PDB names
        print(f"Processing PDB: {name}")
        dssp_tensor = get_dssp_pdb(pdb_path, pdbfasta[name])  # Get DSSP tensor for each PDB
        if dssp_tensor is None:
            fault_name.append(name)  # If failed, add to fault list
        else:
            dssp_tensors.append(dssp_tensor)  # Otherwise, add tensor to list

    if fault_name != []:  # If there are any faults
        print(f"Faulty sequences: {fault_name}")  # Print the faulty sequence names

    return dssp_tensors  # Return all DSSP tensors



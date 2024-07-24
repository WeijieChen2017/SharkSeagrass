import os
import glob
import json
import random

def divide_into_chunks(data, n_chunks):
    """Divide the data into n_chunks."""
    random.shuffle(data)
    return [data[i::n_chunks] for i in range(n_chunks)]

def scan_and_divide_files(save_json_name, target_folder, n_chunks=5):
    # Scan all .nii.gz files
    nii_files = glob.glob(os.path.join(target_folder, '*.nii.gz'))
    print(f"Found {len(nii_files)} .nii.gz files.")
    
    # Dictionary to hold pairs of PET and CT files
    file_pairs = {}
    
    # Group files by their tags
    for file in nii_files:
        filename = os.path.basename(file)
        if '_CT' in filename:
            tag = filename.split('_CT')[0]
            if tag not in file_pairs:
                file_pairs[tag] = {}
            file_pairs[tag]['CT'] = file
        elif '_PET' in filename:
            tag = filename.split('_PET')[0]
            if tag not in file_pairs:
                file_pairs[tag] = {}
            file_pairs[tag]['PET'] = file
    
    # Prepare the list of paired files
    paired_files = []
    for tag, pairs in file_pairs.items():
        if 'PET' in pairs and 'CT' in pairs:
            paired_files.append(pairs)
    
    # Divide the paired files into chunks
    chunks = divide_into_chunks(paired_files, n_chunks)
    
    # Prepare the final dictionary to be saved as JSON
    chunks_dict = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
    
    # Save to JSON
    with open(os.path.join(target_folder, save_json_name), 'w') as json_file:
        json.dump(chunks_dict, json_file, indent=4)
    
    print(f"Chunks saved to {os.path.join(target_folder, save_json_name)}")

# Example usage
target_folder = 'synCT_PET_James'
save_json_name = 'paired_PET_CT_files.json'
scan_and_divide_files(save_json_name, target_folder)

import os
import re

def extract_tag(filename):
    # Extract tag using regex, assuming the tag follows the pattern E#### (e.g., E4055)
    match = re.search(r'E\d{4}', filename)
    if match:
        return match.group(0)
    return None

def scan_and_copy_files(scan_folder, storage_folder):
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)
    
    # Iterate over files in the scan folder
    for root, dirs, files in os.walk(scan_folder):
        for file in files:
            if file.endswith('.nii'):
                # Extract tag from filename
                tag = extract_tag(file)
                if tag:
                    # Construct new filename
                    new_filename = f'PET_{tag}.nii'
                    # Source file path
                    src_path = os.path.join(root, file)
                    # Destination file path
                    dest_path = os.path.join(storage_folder, new_filename)
                    # Copy and rename the file using os.system
                    command = f'cp "{src_path}" "{dest_path}"'
                    print(command)
                    # os.system(command)
                    # print(f'Copied {src_path} to {dest_path}')

# Define the source and destination folders
scan_folder_PET = '/shares/mimrtl/Users/James/synthetic CT PET AC/Duetto_Output_B100_nii'
data_storage_folder = './'

# Run the function
scan_and_copy_files(scan_folder_PET, data_storage_folder)

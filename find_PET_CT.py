import os
import glob
import re
import gzip
import shutil

def extract_tag(filename):
    # Extract tag using regex, assuming the tag follows the pattern E#### (e.g., E4055)
    match = re.search(r'E\d{4}', filename)
    if match:
        return match.group(0)
    return None

def scan_CT_files(CT_reference_folder):
    CT_file_list = sorted(glob.glob(os.path.join(CT_reference_folder, '*/*.nii')))
    for CT_file in CT_file_list:
        print(f'Found CT file: {CT_file}')
    return CT_file_list

def scan_and_copy_files(scan_folder, storage_folder, CT_file_list):
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)
    # Iterate over files in the scan folder
    for root, dirs, files in os.walk(scan_folder):
        for file in files:
            if file.endswith('.nii'):
                # Extract tag from filename
                tag = extract_tag(file)
                if tag:
                    tag = tag[1:]
                    # # Construct new filename
                    # new_filename = f'PET_{tag}.nii'
                    # # Source file path
                    # src_path = os.path.join(root, file)
                    # # Destination file path
                    # dest_path = os.path.join(storage_folder, new_filename)
                    # # Copy and rename the file using os.system
                    # command = f'cp "{src_path}" "{dest_path}"'
                    # # print(command)
                    # os.system(command)
                    # print(f'Copied {src_path} to {dest_path}')
                    print(f"{tag} found.")
                    for CT_file_path in CT_file_list:
                        if f"_{tag}_" in CT_file_path and not "Eq" in CT_file_path:
                            # print(f"Found CT file for {tag} at {CT_file_path}")
                            # Construct new filename
                            new_filename = os.path.join(storage_folder, f"CT_E{tag}.nii")
                            command = f'cp "{CT_file_path}" "{new_filename}"'
                            print(command)
                            os.system(command)
                            print(f"CT file for {tag} copied to {storage_folder}")

def rearrange_files(storage_folder):
    for root, dirs, files in os.walk(storage_folder):
        for file in files:
            if file.endswith('.nii'):
                match = re.match(r'(CT|PET)_E(\d{4})\.nii', file)
                if match:
                    modality = match.group(1)
                    tag = match.group(2)
                    new_filename = f'E{tag}_{modality}.nii'
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(root, new_filename)
                    os.rename(src_path, dest_path)
                    print(f'Renamed {src_path} to {dest_path}')

def compress_nii_files(storage_folder):
    for root, dirs, files in os.walk(storage_folder):
        for file in files:
            if file.endswith('.nii'):
                nii_path = os.path.join(root, file)
                nii_gz_path = nii_path + '.gz'
                
                # Compress the file
                with open(nii_path, 'rb') as f_in:
                    with gzip.open(nii_gz_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f'Compressed {nii_path} to {nii_gz_path}')
                
                # Optionally, remove the original .nii file
                os.remove(nii_path)
                print(f'Removed original file {nii_path}')
    


# Define the source and destination folders
scan_folder_PET = '/shares/mimrtl/Users/James/synthetic CT PET AC/Duetto_Output_B100_nii'
CT_reference_folder = '/shares/mimrtl/Users/James/synthetic CT PET AC/CTAC_nii'
data_storage_folder = '.'
# Run the function
CT_file_list = scan_CT_files(CT_reference_folder)
scan_and_copy_files(scan_folder_PET, data_storage_folder, CT_file_list)
rearrange_files(data_storage_folder)
compress_nii_files(data_storage_folder)
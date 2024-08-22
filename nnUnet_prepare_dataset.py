import os
import shutil

# Paths to your data directories
data_path = "./nnUnet/nnUNet_raw_data_base/Task001_baseline"
pet_path = data_path  # PET_TOFNAC files are in this directory
ctac_path = data_path  # CTACIVV files are in this directory

# Destination directories
imagesTr = os.path.join(data_path, "imagesTr")
labelsTr = os.path.join(data_path, "labelsTr")

# Create directories if they don't exist
os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)

# Loop over PET files (input images)
for pet_file in os.listdir(pet_path):
    if pet_file.startswith("PET_TOFNAC"):
        # Extract the unique identifier (e.g., E4055)
        case_id = pet_file.split('_')[2].replace('.nii.gz', '')
        
        # Rename and move PET files to imagesTr
        pet_new_name = f"{case_id}_0000.nii.gz"
        shutil.move(os.path.join(pet_path, pet_file), os.path.join(imagesTr, pet_new_name))
        
        # Find the corresponding CTACIVV file (label)
        ctac_file = f"CTACIVV_{case_id}.nii.gz"
        
        if os.path.exists(os.path.join(ctac_path, ctac_file)):
            # Rename and move CTACIVV files to labelsTr
            shutil.move(os.path.join(ctac_path, ctac_file), os.path.join(labelsTr, f"{case_id}.nii.gz"))
        else:
            print(f"Warning: No corresponding CTACIVV file found for {pet_file}")

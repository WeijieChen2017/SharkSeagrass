#!/bin/bash
# example run: ./petdcm2nii.sh Duetto_Output_B50


# Check if the folder name is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name>"
  exit 1
fi

# Assign the folder name to a variable
FOLDER_NAME=$1

# Check if the folder exists
if [ ! -d "$FOLDER_NAME" ]; then
  echo "Folder '$FOLDER_NAME' does not exist."
  exit 1
fi

# Specify the target folder containing DICOM files
output_root="${FOLDER_NAME}_nii_Winston"

# Specify the root directory containing your data
root_dir=$FOLDER_NAME

# Find all subdirectories containing DICOM files and loop through them
find "$root_dir" -type d | while read -r folder; do
    # Check if the folder contains DICOM files (you can adjust the pattern if needed)
    if ls "$folder"/*.sdcopen 1> /dev/null 2>&1; then
        # Extract the folder name
        folder_name=$(basename "$folder")
        
        # Specify the output directory for NIfTI files
        output_dir="$output_root/$folder_name"

        # Create the output directory if it doesn't exist
        mkdir -p "$output_dir"

        # Use dcm2niix to convert DICOMs to NIfTI in the output directory
        dcm2niix -o "$output_dir" "$folder"
    fi
done

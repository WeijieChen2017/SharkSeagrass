import glob
import os

# file_list = ["E4063_CT.nii.gz"]

file_list = sorted(glob.glob("*.nii.gz"))

for filepath in file_list:
    # print(filepath)
    # filename = os.path.basename(filepath)
    # new_filename = filename.replace(".nii.gz", "_400.nii.gz")
    # cmd = f"3dresample -dxyz 1.5 1.5 1.5 -rmode Cu -prefix {new_filename} -input {filename}"
    # print(cmd)
    # os.system(cmd)
    # print(f"---Resampled data saved at {new_filename}")

    # filename = os.path.basename(filepath)
    # new_filename = filename.replace(".nii.gz", "_256.nii.gz")
    # cmd = f"3dresample -dxyz 2.344 2.344 2.344 -rmode Cu -prefix {new_filename} -input {filename}"
    # print(cmd)

    filename = os.path.basename(filepath)
    new_filename = filename.replace(".nii.gz", "_oriCTAC.nii.gz")
    cmd = f"3dresample -dxyz 1.367 1.367 3.27 -rmode Cu -prefix {new_filename} -input {filename}"
    print(cmd)
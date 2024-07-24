import glob
import os

file_list = glob.glob("*.nii.gz")
for filepath in file_list:
    print(filepath)
    filename = os.path.basename(filepath)
    new_filename = filename.replace(".nii.gz", "_re.nii.gz")
    cmd = f"3dresample -dxyz 1.5 1.5 1.5 -rmode Cu -prefix {new_filename} -input {filename}"
    print(cmd)
    os.system(cmd)
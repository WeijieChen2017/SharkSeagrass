data_folder = f"B100/TOFNAC_CTACIVV_part2/"
data_div_json = "UNetUNet_v1_data_split.json"

import json
import os
import nibabel as nib

with open(data_div_json, "r") as f:
    data_div = json.load(f)

data_div_cv = data_div[f"cv_0"]

for split in ["train", "valid", "test"]:
    print(f"{split}: {len(data_div_cv[split])}")
    for casename in data_div_cv[split]:
        print(f"  {casename}")
        CTAC_path = os.path.join(data_folder, f"{casename}_CTAC.nii.gz")
        CTAC_file = nib.load(CTAC_path)
        CTAC_data = CTAC_file.get_fdata()
        print(f"    {CTAC_data.shape}")


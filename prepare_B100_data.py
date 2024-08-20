tag_list = [
    "E4055", "E4058", "E4061",          "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079",          "E4081", "E4084",
             "E4091", "E4092", "E4094", "E4096",
             "E4098", "E4099",          "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
]

import os

for tag in tag_list:
    CT_path = f"./B100/CTACIVV/CTACIVV_{tag[1:]}.nii.gz"
    PET_path = f"./B100/TOFNAC/PET_TOFNAC_{tag}.nii.gz"
    
    dst_CT_path = f"./B100/CTACIVV_resample/CTACIVV_{tag[1:]}.nii.gz"
    dst_PET_path = f"./B100/TOFNAC_resample/PET_TOFNAC_{tag}.nii.gz"

    CT_cmd = f"3dresample -dxyz 1.5 1.5 1.5 -prefix {dst_CT_path} -inset {CT_path}"
    PET_cmd = f"3dresample -dxyz 1.5 1.5 1.5 -prefix {dst_PET_path} -inset {PET_path}"

    print(CT_cmd)
    print(PET_cmd)

    os.system(CT_cmd)
    os.system(PET_cmd)

    print("<===============================================>")
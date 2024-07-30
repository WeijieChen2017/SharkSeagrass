import os


# tag_list = ["E4055", "E4058", "E4061", "E4063", "E4066",
#             "E4068", "E4069", "E4073", "E4074", "E4077",
#             "E4078", "E4079", "E4080", "E4081", "E4084",
#             "E4087", "E4091", "E4092", "E4094"]

tag_list = ["E4055", "E4058", "E4061",          "E4066",
            "E4068", "E4069", "E4073", "E4074", "E4077",
            "E4078", "E4079",          "E4081", "E4084",
                     "E4091", "E4092", "E4094"]

target_dir = f"synCT_PET_James/ori/crop/"

for tag in tag_list:

    # ---Cropped CT data saved at synCT_PET_James/ori/E4081_CT_crop_th04.nii.gz
    # ---Cropped PET data saved at synCT_PET_James/ori/E4081_PET_crop_th04.nii.gz
    # ---Cropped PET mask saved at synCT_PET_James/ori/E4081_PET_mask_crop_th04.nii.gz
    # ---Cropped smoothed PET data saved at synCT_PET_James/ori/E4081_PET_GauKer3_crop_th04.nii.gz
    # ---Cropped smoothed gradient magnitude PET data saved at synCT_PET_James/ori/E4081_PET_GradMag_crop_th04.nii.gz

    ct_path = f"synCT_PET_James/ori/{tag}_CT_crop_th04.nii.gz"
    pet_path = f"synCT_PET_James/ori/{tag}_PET_crop_th04.nii.gz"
    pet_mask_path = f"synCT_PET_James/ori/{tag}_PET_mask_crop_th04.nii.gz"
    pet_smooth_path = f"synCT_PET_James/ori/{tag}_PET_GauKer3_crop_th04.nii.gz"
    pet_grad_path = f"synCT_PET_James/ori/{tag}_PET_GradMag_crop_th04.nii.gz"

    cmd_ct = f"cp {ct_path} {target_dir}"
    cmd_pet = f"cp {pet_path} {target_dir}"
    cmd_pet_mask = f"cp {pet_mask_path} {target_dir}"
    cmd_pet_smooth = f"cp {pet_smooth_path} {target_dir}"
    cmd_pet_grad = f"cp {pet_grad_path} {target_dir}"

    os.system(cmd_ct)
    os.system(cmd_pet)
    os.system(cmd_pet_mask)
    os.system(cmd_pet_smooth)
    os.system(cmd_pet_grad)

    print(f"---{ct_path} copied to {target_dir}")
    print(f"---{pet_path} copied to {target_dir}")
    print(f"---{pet_mask_path} copied to {target_dir}")
    print(f"---{pet_smooth_path} copied to {target_dir}")
    print(f"---{pet_grad_path} copied to {target_dir}")

    
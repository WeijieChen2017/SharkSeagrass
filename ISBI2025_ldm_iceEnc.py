# cv_list = ["cv1", "cv3", "cv4"] # for ldm
cv_list = ["cv0", "cv1", "cv2", "cv3", "cv4"] # for iceEnc
# cv_list = ["cv0", "cv1"]

split_list = ["test"]

# data_fusion_list = ["axial", "sagittal", "coronal", "average", "median"]
data_fusion_list = ["median"]

# region_list = ["whole", "air", "soft", "bone"]
region_list = ["whole", "soft", "bone"]

# CT unit is HU
# PET unit is Bq/ml

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes
import json

# compute metrics including MAE, PSNR, SSIM, DSC
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


WRONG_MAX_CT = 2976
CORRECT_MAX_CT = 976
SCRATCH_MAX_CT = 2976
MIN_CT = -1024
WRONG_CT_RANGE = WRONG_MAX_CT - MIN_CT
CORRECT_CT_RANGE = CORRECT_MAX_CT - MIN_CT
SCRATCH_CT_RANGE = SCRATCH_MAX_CT - MIN_CT

CT_mask_folder = "ISBI2025_mask/"
os.makedirs(CT_mask_folder, exist_ok=True)
HU_boundary_valid_air = -450
# HU_boundary_air_soft = -250
# HU_boundary_soft_bone = 150
HU_boundary_air = [-1024, -450]
HU_boundary_soft = [-450, 150]
HU_boundary_bone = [150, 2976]

# HU_boundary_valid_air = -450
# HU_boundary_soft = [-450, 150]
# HU_boundary_valid_bone = 150

for cv in cv_list:
    data_div_json = "UNetUNet_v1_data_split.json" # use this for now
    with open(data_div_json, "r") as f:
        data_div_dict = json.load(f)
    split_dict = data_div_dict[cv.replace("cv", "cv_")]


    for split in split_list:
        print("Processing split: ", split)
        metrics_dict = {}
        for region in region_list:
            for data_fusion in data_fusion_list:
                metrics_dict[f"synCT_MAE_{region}_{data_fusion}"] = []
                metrics_dict[f"synCT_PSNR_{region}_{data_fusion}"] = []
                metrics_dict[f"synCT_SSIM_{region}_{data_fusion}"] = []
                metrics_dict[f"synCT_DSC_{region}_{data_fusion}"] = []
        result_save_json = f"ISBI2025_ldm_iceEnc_metrics_{cv}_{split}_updatedMask.json"
        casename_list = sorted(split_dict[split])
        pred_folder = f"results/{cv}_256_iceEnc/{split}/"

        # this is 400*400, we need generate 256*256 mask first, then compute it.
        for casename in casename_list:
            E_casename = "E4"+casename[3:]
            # determine whether it is part 1 or part 2
            part_1_or_part_2 = None
            case_index = int(casename[3:])
            if case_index <= 139:
                part_1_or_part_2 = "part1"
            else:
                part_1_or_part_2 = "part2"
                continue # we did not eval part 2 for now
            
            # prepare the ground truth CT data
            CT_GT_path = f"TC256_v2/{casename}_CTAC_256.nii.gz"
            CT_GT_correct_path = CT_GT_path.replace(".nii.gz", "_corrected.nii.gz")
            if os.path.exists(CT_GT_correct_path):
                CT_GT_file = nib.load(CT_GT_correct_path)
                CT_GT_data = CT_GT_file.get_fdata()
                # print("Loaded corrected CT_GT from: ", CT_GT_correct_path)
            else:
                CT_GT_file = nib.load(CT_GT_path)
                CT_GT_data = CT_GT_file.get_fdata()
                CT_GT_data = np.clip(CT_GT_data, 0, 1)
                if part_1_or_part_2 == "part1":
                    CT_GT_data = CT_GT_data * WRONG_CT_RANGE + MIN_CT
                elif part_1_or_part_2 == "part2":
                    CT_GT_data = CT_GT_data * CORRECT_CT_RANGE + MIN_CT
                else:
                    raise ValueError("Invalid part_1_or_part_2")
                # save the corrected CT_GT_data
                CT_GT_correct_file = nib.Nifti1Image(CT_GT_data, CT_GT_file.affine, CT_GT_file.header)
                nib.save(CT_GT_correct_file, CT_GT_correct_path)
                # print("Saved corrected CT_GT to: ", CT_GT_correct_path)
            
            # prepare the CT mask
            mask_CT_whole_path = os.path.join(CT_mask_folder, f"CT_mask_{casename}.nii.gz")
            mask_CT_air_path = os.path.join(CT_mask_folder, f"CT_mask_air_{casename}.nii.gz")
            mask_CT_soft_path = os.path.join(CT_mask_folder, f"CT_mask_soft_{casename}.nii.gz")
            mask_CT_bone_path = os.path.join(CT_mask_folder, f"CT_mask_bone_{casename}.nii.gz")

            if os.path.exists(mask_CT_whole_path):
                mask_CT_whole_file = nib.load(mask_CT_whole_path)
                mask_CT_whole = mask_CT_whole_file.get_fdata()
                mask_CT_whole = mask_CT_whole > 0

                mask_CT_air_file = nib.load(mask_CT_air_path)
                mask_CT_air = mask_CT_air_file.get_fdata()
                mask_CT_air = mask_CT_air > 0

                mask_CT_soft_file = nib.load(mask_CT_soft_path)
                mask_CT_soft = mask_CT_soft_file.get_fdata()
                mask_CT_soft = mask_CT_soft > 0

                mask_CT_bone_file = nib.load(mask_CT_bone_path)
                mask_CT_bone = mask_CT_bone_file.get_fdata()
                mask_CT_bone = mask_CT_bone > 0

                # print("Loaded masks for whole, air, soft, bone from: ", CT_mask_folder)
            else:
                mask_CT_whole = CT_GT_data > HU_boundary_valid_air
                for i in range(CT_GT_data.shape[2]):
                    mask_CT_whole[:, :, i] = binary_fill_holes(mask_CT_whole[:, :, i])
                
                # save the mask_CT_whole
                mask_CT_whole_file = nib.Nifti1Image(mask_CT_whole.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                nib.save(mask_CT_whole_file, mask_CT_whole_path)
                # print("Saved whole mask to: ", mask_CT_whole_path)
                
                # air mask is from MIN to HU_boundary_air_soft
                mask_CT_air = (CT_GT_data >= HU_boundary_air[0]) & (CT_GT_data <= HU_boundary_air[1])
                # intersection with the whole mask
                mask_CT_air = mask_CT_air & mask_CT_whole
                # save the mask
                mask_CT_air_file = nib.Nifti1Image(mask_CT_air.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                nib.save(mask_CT_air_file, mask_CT_air_path)
                # print("Saved air mask to: ", mask_CT_air_path)

                # soft mask is from HU_boundary_air_soft to HU_boundary_soft_bone
                mask_CT_soft = (CT_GT_data >= HU_boundary_soft[0]) & (CT_GT_data <= HU_boundary_soft[1])
                # intersection with the whole mask
                mask_CT_soft = mask_CT_soft & mask_CT_whole
                # save the mask
                mask_CT_soft_file = nib.Nifti1Image(mask_CT_soft.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                nib.save(mask_CT_soft_file, mask_CT_soft_path)
                # print("Saved soft mask to: ", mask_CT_soft_path)

                # bone mask is from HU_boundary_soft_bone to MAX
                mask_CT_bone = (CT_GT_data >= HU_boundary_bone[0]) & (CT_GT_data <= HU_boundary_bone[1])
                # intersection with the whole mask
                mask_CT_bone = mask_CT_bone & mask_CT_whole
                # save the mask
                mask_CT_bone_file = nib.Nifti1Image(mask_CT_bone.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                nib.save(mask_CT_bone_file, mask_CT_bone_path)
                # print("Saved bone mask to: ", mask_CT_bone_path)

            # prepare the predicted CT data
            for data_fusion in data_fusion_list:
                pred_path = pred_folder+f"{casename}_CTAC_pred_{data_fusion}_{cv}.nii.gz"
                pred_correct_path = pred_path.replace(".nii.gz", "_corrected.nii.gz")
                if os.path.exists(pred_correct_path):
                    pred_correct_file = nib.load(pred_correct_path)
                    pred_data_correct = pred_correct_file.get_fdata()
                    # print("Loaded corrected pred from: ", pred_correct_path)
                else:
                    pred_file = nib.load(pred_path)
                    pred_data = pred_file.get_fdata()
                    # pred_data_norm = (pred_data - MIN_CT) / CORRECT_CT_RANGE
                    # pred_data_correct = pred_data_norm * WRONG_CT_RANGE + MIN_CT
                    # here pred_data is from -1024 to some value, and the zero points are near -550
                    pred_data_norm = (pred_data - MIN_CT) / CORRECT_CT_RANGE
                    pred_data_correct = pred_data_norm * SCRATCH_CT_RANGE + MIN_CT
                    # save the corrected pred_data
                    pred_correct_file = nib.Nifti1Image(pred_data_correct, pred_file.affine, pred_file.header)
                    nib.save(pred_correct_file, pred_correct_path)
                    # print("Saved corrected pred to: ", pred_correct_path)
                
                # check whether the third dimension is the same
                if pred_data_correct.shape[2] != mask_CT_whole.shape[2]:
                    pred_data_correct = pred_data_correct[:, :, :mask_CT_whole.shape[2]]

                # compute the predicted data mask
                mask_CT_whole_pred = pred_data_correct > -500
                for i in range(pred_data_correct.shape[2]):
                    mask_CT_whole_pred[:, :, i] = binary_fill_holes(mask_CT_whole_pred[:, :, i])
                
                pred_mask_dict = {
                    "whole": mask_CT_whole_pred,
                    "air": (pred_data_correct >= HU_boundary_air[0]) & (pred_data_correct <= HU_boundary_air[1]),
                    "soft": (pred_data_correct >= HU_boundary_soft[0]) & (pred_data_correct <= HU_boundary_soft[1]),
                    "bone": (pred_data_correct >= HU_boundary_bone[0]) & (pred_data_correct <= HU_boundary_bone[1])
                }
                # save the predicted mask
                # pred_mask_whole_file = nib.Nifti1Image(pred_mask_dict["whole"].astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                # nib.save(pred_mask_whole_file, os.path.join(CT_mask_folder, f"pred_mask_whole_{casename}_{data_fusion}_{cv}.nii.gz"))
                # pred_mask_air_file = nib.Nifti1Image(pred_mask_dict["air"].astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                # nib.save(pred_mask_air_file, os.path.join(CT_mask_folder, f"pred_mask_air_{casename}_{data_fusion}_{cv}.nii.gz"))
                # pred_mask_soft_file = nib.Nifti1Image(pred_mask_dict["soft"].astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                # nib.save(pred_mask_soft_file, os.path.join(CT_mask_folder, f"pred_mask_soft_{casename}_{data_fusion}_{cv}.nii.gz"))
                # pred_mask_bone_file = nib.Nifti1Image(pred_mask_dict["bone"].astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
                # nib.save(pred_mask_bone_file, os.path.join(CT_mask_folder, f"pred_mask_bone_{casename}_{data_fusion}_{cv}.nii.gz"))

                # compute the metrics
                for region in region_list:
                    if region == "whole":
                        mask = mask_CT_whole
                    elif region == "air":
                        mask = mask_CT_air
                    elif region == "soft":
                        mask = mask_CT_soft
                    elif region == "bone":
                        mask = mask_CT_bone
                    else:
                        raise ValueError("Invalid region")

                    MAE = np.mean(np.abs(CT_GT_data[mask] - pred_data_correct[mask]))
                    metrics_dict[f"synCT_MAE_{region}_{data_fusion}"].append(MAE)
                    # print(f"Case {casename}, split {split}, synCT_MAE_{region}_{data_fusion}: ", MAE)
                    print(f"{MAE:.4f}")

                    # compute psnr
                    postive_CT_GT_data = CT_GT_data- MIN_CT
                    postive_pred_data_correct = pred_data_correct - MIN_CT
                    PSNR = psnr(postive_CT_GT_data[mask], postive_pred_data_correct[mask], data_range=SCRATCH_CT_RANGE)
                    metrics_dict[f"synCT_PSNR_{region}_{data_fusion}"].append(PSNR)
                    # print(f"Case {casename}, split {split}, synCT_PSNR_{region}_{data_fusion}: ", PSNR)
                    print(f"{PSNR:.4f}")

                    # compute ssim
                    SSIM = ssim(postive_CT_GT_data[mask], postive_pred_data_correct[mask], data_range=SCRATCH_CT_RANGE)
                    metrics_dict[f"synCT_SSIM_{region}_{data_fusion}"].append(SSIM)
                    # print(f"Case {casename}, split {split}, synCT_SSIM_{region}_{data_fusion}: ", SSIM)
                    print(f"{SSIM:.4f}")

                    # compute dice coefficient
                    GT_mask = mask
                    pred_mask = pred_mask_dict[region]
                    intersection = np.sum(GT_mask & pred_mask)
                    union = np.sum(GT_mask) + np.sum(pred_mask)
                    DSC = 2 * intersection / union
                    metrics_dict[f"synCT_DSC_{region}_{data_fusion}"].append(DSC)
                    # print(f"Case {casename}, split {split}, synCT_DSC_{region}_{data_fusion}: ", DSC)
                    print(f"{DSC:.4f}")
                    print()


        for key in metrics_dict.keys():
            metrics_dict[key] = np.mean(metrics_dict[key])
        
        # in json, output metric names first per row
        with open(result_save_json, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # print("Saved metrics to: ", result_save_json)
        # print("Metrics: ", metrics_dict)
    #     print(">"*50)
    
    # print(">"*50)
    # print()

    #     # CT data is from -1 to 1
    #     # PET data is from -1 to 1
    #     # scale the data to the original range
    #     CT_data_denorm = (CT_data + 1) / 2 * RANGE_CT + MIN_CT
    #     PET_data_denorm = (PET_data + 1) / 2
    #     PET_data_denorm = reverse_two_segment_scale(PET_data_denorm, MIN_PET, MID_PET, MAX_PET, MIQ_PET)

    #     CT_GT_path = f"B100/CTACIVV_resample/CTACIVV_{casename[1:]}.nii.gz"
    #     PET_GT_path = f"B100/TOFNAC_resample/PET_TOFNAC_{casename}.nii.gz"

    #     CT_GT_file = nib.load(CT_GT_path)
    #     PET_GT_file = nib.load(PET_GT_path)

    #     CT_GT_data = CT_GT_file.get_fdata()[33:433, 33:433, :]
    #     PET_GT_data = PET_GT_file.get_fdata()

    #     # clip the data to the original range
    #     CT_GT_data = np.clip(CT_GT_data, MIN_CT, MAX_CT)
    #     PET_GT_data = np.clip(PET_GT_data, MIN_PET, MAX_PET)
    #     CT_data_denorm = np.clip(CT_data_denorm, MIN_CT, MAX_CT)
    #     PET_data_denorm = np.clip(PET_data_denorm, MIN_PET, MAX_PET)

    #     # compute the mask using CT_GT_data if the mask does not exist
    #     mask_CT_whole_path = os.path.join(CT_mask_folder, f"CT_mask_{casename}.nii.gz")
    #     mask_CT_air_path = os.path.join(CT_mask_folder, f"CT_mask_air_{casename}.nii.gz")
    #     mask_CT_soft_path = os.path.join(CT_mask_folder, f"CT_mask_soft_{casename}.nii.gz")
    #     mask_CT_bone_path = os.path.join(CT_mask_folder, f"CT_mask_bone_{casename}.nii.gz")

    #     if os.path.exists(mask_CT_whole_path):
    #         mask_CT_whole_file = nib.load(mask_CT_whole_path)
    #         mask_CT_whole = mask_CT_whole_file.get_fdata()
    #         mask_CT_whole = mask_CT_whole > 0

    #         mask_CT_air_file = nib.load(mask_CT_air_path)
    #         mask_CT_air = mask_CT_air_file.get_fdata()
    #         mask_CT_air = mask_CT_air > 0

    #         mask_CT_soft_file = nib.load(mask_CT_soft_path)
    #         mask_CT_soft = mask_CT_soft_file.get_fdata()
    #         mask_CT_soft = mask_CT_soft > 0

    #         mask_CT_bone_file = nib.load(mask_CT_bone_path)
    #         mask_CT_bone = mask_CT_bone_file.get_fdata()
    #         mask_CT_bone = mask_CT_bone > 0
    #     else:
    #         mask_CT_whole = CT_GT_data > -500
    #         for i in range(CT_GT_data.shape[2]):
    #             mask_CT_whole[:, :, i] = binary_fill_holes(mask_CT_whole[:, :, i])
            
    #         # save the mask_CT_whole
    #         mask_CT_whole_file = nib.Nifti1Image(mask_CT_whole.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
    #         nib.save(mask_CT_whole_file, mask_CT_whole_path)
    #         print("Saved whole mask to: ", mask_CT_whole_path)
            
    #         # air mask is from MIN to HU_boundary_air_soft
    #         mask_CT_air = (CT_GT_data > MIN_CT) & (CT_GT_data < HU_boundary_air_soft)
    #         # intersection with the whole mask
    #         mask_CT_air = mask_CT_air & mask_CT_whole
    #         # save the mask
    #         mask_CT_air_file = nib.Nifti1Image(mask_CT_air.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
    #         nib.save(mask_CT_air_file, mask_CT_air_path)
    #         print("Saved air mask to: ", mask_CT_air_path)

    #         # soft mask is from HU_boundary_air_soft to HU_boundary_soft_bone
    #         mask_CT_soft = (CT_GT_data > HU_boundary_air_soft) & (CT_GT_data < HU_boundary_soft_bone)
    #         # intersection with the whole mask
    #         mask_CT_soft = mask_CT_soft & mask_CT_whole
    #         # save the mask
    #         mask_CT_soft_file = nib.Nifti1Image(mask_CT_soft.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
    #         nib.save(mask_CT_soft_file, mask_CT_soft_path)
    #         print("Saved soft mask to: ", mask_CT_soft_path)

    #         # bone mask is from HU_boundary_soft_bone to MAX
    #         mask_CT_bone = (CT_GT_data > HU_boundary_soft_bone) & (CT_GT_data < MAX_CT)
    #         # intersection with the whole mask
    #         mask_CT_bone = mask_CT_bone & mask_CT_whole
    #         # save the mask
    #         mask_CT_bone_file = nib.Nifti1Image(mask_CT_bone.astype(np.float32), CT_GT_file.affine, CT_GT_file.header)
    #         nib.save(mask_CT_bone_file, mask_CT_bone_path)
    #         print("Saved bone mask to: ", mask_CT_bone_path)

    #     # start to compute the metrics
    #     # compute the metrics for the whole mask
    #     CT_MAE_whole = np.mean(np.abs(CT_GT_data[mask_CT_whole] - CT_data_denorm[mask_CT_whole]))
    #     PET_MAE_whole = np.mean(np.abs(PET_GT_data[mask_CT_whole] - PET_data_denorm[mask_CT_whole]))
    #     CT_MAE_air = np.mean(np.abs(CT_GT_data[mask_CT_air] - CT_data_denorm[mask_CT_air]))
    #     PET_MAE_air = np.mean(np.abs(PET_GT_data[mask_CT_air] - PET_data_denorm[mask_CT_air]))
    #     CT_MAE_soft = np.mean(np.abs(CT_GT_data[mask_CT_soft] - CT_data_denorm[mask_CT_soft]))
    #     PET_MAE_soft = np.mean(np.abs(PET_GT_data[mask_CT_soft] - PET_data_denorm[mask_CT_soft]))
    #     CT_MAE_bone = np.mean(np.abs(CT_GT_data[mask_CT_bone] - CT_data_denorm[mask_CT_bone]))
    #     PET_MAE_bone = np.mean(np.abs(PET_GT_data[mask_CT_bone] - PET_data_denorm[mask_CT_bone]))

    #     metrics_dict["CT_MAE_whole"].append(CT_MAE_whole)
    #     metrics_dict["PET_MAE_whole"].append(PET_MAE_whole)
    #     metrics_dict["CT_MAE_air"].append(CT_MAE_air)
    #     metrics_dict["PET_MAE_air"].append(PET_MAE_air)
    #     metrics_dict["CT_MAE_soft"].append(CT_MAE_soft)
    #     metrics_dict["PET_MAE_soft"].append(PET_MAE_soft)
    #     metrics_dict["CT_MAE_bone"].append(CT_MAE_bone)
    #     metrics_dict["PET_MAE_bone"].append(PET_MAE_bone)

    #     print("CT_MAE_whole: ", CT_MAE_whole)
    #     print("PET_MAE_whole: ", PET_MAE_whole)
    #     print("CT_MAE_air: ", CT_MAE_air)
    #     print("PET_MAE_air: ", PET_MAE_air)
    #     print("CT_MAE_soft: ", CT_MAE_soft)
    #     print("PET_MAE_soft: ", PET_MAE_soft)
    #     print("CT_MAE_bone: ", CT_MAE_bone)
    #     print("PET_MAE_bone: ", PET_MAE_bone)
    #     print("<"*25)

    # # save the dict
    # metric_dict_name = f"ISBI2025_ldm_recon_metrics_dict_{model_spec}.npy"
    # np.save(metric_dict_name, metrics_dict)
    # print("Saved metrics dict to: ", metric_dict_name)

    # for key in metrics_dict.keys():
    #     metrics_dict[key] = np.mean(metrics_dict[key])
    
    # # in json, output metric names first per row
    # with open(result_save_json, "w") as f:
    #     json.dump(metrics_dict, f, indent=4)
    # print("Saved metrics to: ", result_save_json)

    # print("Metrics: ", metrics_dict)
    # print(">"*50)
    # print()

 


        






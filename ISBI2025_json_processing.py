cv0 = {
    "synCT_MAE_whole_median": 72.61356863312905,
    "synCT_PSNR_whole_median": 28.93674539192831,
    "synCT_SSIM_whole_median": 0.8826097191084914,
    "synCT_DSC_whole_median": 0.9843046395908962,
    "synCT_MAE_soft_median": 53.641046691493514,
    "synCT_PSNR_soft_median": 32.11898114590366,
    "synCT_SSIM_soft_median": 0.9191083299784061,
    "synCT_DSC_soft_median": 0.9523145441312987,
    "synCT_MAE_bone_median": 216.992588037729,
    "synCT_PSNR_bone_median": 22.550708554765826,
    "synCT_SSIM_bone_median": 0.6394101593279266,
    "synCT_DSC_bone_median": 0.6451112804049739
}

cv1 = {
    "synCT_MAE_whole_median": 83.77747990267706,
    "synCT_PSNR_whole_median": 27.885880290680003,
    "synCT_SSIM_whole_median": 0.8729778086947091,
    "synCT_DSC_whole_median": 0.9769989650084864,
    "synCT_MAE_soft_median": 61.352418667563335,
    "synCT_PSNR_soft_median": 30.87504071468515,
    "synCT_SSIM_soft_median": 0.9145062446899541,
    "synCT_DSC_soft_median": 0.9459423011110345,
    "synCT_MAE_bone_median": 212.50322259584132,
    "synCT_PSNR_bone_median": 22.48692803050012,
    "synCT_SSIM_bone_median": 0.6222345456182513,
    "synCT_DSC_bone_median": 0.6909842177924871
}

cv2 = {
    "synCT_MAE_whole_median": 76.9100387518142,
    "synCT_PSNR_whole_median": 28.171964832443887,
    "synCT_SSIM_whole_median": 0.8786315876725934,
    "synCT_DSC_whole_median": 0.9711820837681685,
    "synCT_MAE_soft_median": 56.596664972088895,
    "synCT_PSNR_soft_median": 30.649363542442433,
    "synCT_SSIM_soft_median": 0.9169558425054112,
    "synCT_DSC_soft_median": 0.9477249275389237,
    "synCT_MAE_bone_median": 280.37300957649046,
    "synCT_PSNR_bone_median": 21.15020209280811,
    "synCT_SSIM_bone_median": 0.6168513268765815,
    "synCT_DSC_bone_median": 0.5519469493081915
}

cv3 = {
    "synCT_MAE_whole_median": 70.21552814258663,
    "synCT_PSNR_whole_median": 28.951998778877794,
    "synCT_SSIM_whole_median": 0.88263397433409,
    "synCT_DSC_whole_median": 0.9862925484305537,
    "synCT_MAE_soft_median": 52.90256311601253,
    "synCT_PSNR_soft_median": 31.703775279799434,
    "synCT_SSIM_soft_median": 0.917502533225239,
    "synCT_DSC_soft_median": 0.9532639398403692,
    "synCT_MAE_bone_median": 216.36721189785663,
    "synCT_PSNR_bone_median": 22.37801014947583,
    "synCT_SSIM_bone_median": 0.6029494803888825,
    "synCT_DSC_bone_median": 0.6290630122414755
}

cv4 = {
    "synCT_MAE_whole_median": 68.92183368607573,
    "synCT_PSNR_whole_median": 29.36570210468054,
    "synCT_SSIM_whole_median": 0.882009042157147,
    "synCT_DSC_whole_median": 0.9872599565252823,
    "synCT_MAE_soft_median": 44.63827868407436,
    "synCT_PSNR_soft_median": 33.01632238013476,
    "synCT_SSIM_soft_median": 0.9249581857248419,
    "synCT_DSC_soft_median": 0.950502583412922,
    "synCT_MAE_bone_median": 215.74235488466067,
    "synCT_PSNR_bone_median": 22.92814351193343,
    "synCT_SSIM_bone_median": 0.6181353781033323,
    "synCT_DSC_bone_median": 0.6322283507850084
}

# take the mean of the 5 folds

key_list = list(cv0.keys())
cv_mean = {}
for key in key_list:
    cv_mean[key] = (cv0[key] + cv1[key] + cv2[key] + cv3[key] + cv4[key]) / 5

for key in key_list:
    print(f"{cv_mean[key]:.3f}")
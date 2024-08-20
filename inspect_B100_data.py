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

# here we load the data
import os
import nibabel as nib
import numpy as np

for tag in tag_list:
    CT_path = f"./B100/CTACIVV_resample/CTACIVV_{tag[1:]}.nii.gz"
    PET_path = f"./B100/TOFNAC_resample/PET_TOFNAC_{tag}.nii.gz"
    CT_file = nib.load(CT_path)
    PET_file = nib.load(PET_path)
    CT_data = CT_file.get_fdata()
    PET_data = PET_file.get_fdata()

    print("<"*50)
    print(f"File: CTACIVV_{tag[1:]}.nii.gz")
    print(f"CT shape: {CT_data.shape}, CT_max: {np.max(CT_data)}, CT_min: {np.min(CT_data)}")
    print(f"CT mean: {np.mean(CT_data):.4f}, CT std: {np.std(CT_data):.4f}")
    print(f"CT 95th, 99th percentile: {np.percentile(CT_data, 95):.4f} {np.percentile(CT_data, 99):.4f}")
    print(f"CT 99.9th, 99.99th percentile: {np.percentile(CT_data, 99.9):.4f}, {np.percentile(CT_data, 99.99):.4f}")
    dx, dy, dz = CT_file.header.get_zooms()
    print(f"CT physcial spacing: {dx:.4f} {dy:.4f} {dz:.4f}, range: {dx * CT_data.shape[0]:.4f} {dy * CT_data.shape[1]:.4f} {dz * CT_data.shape[2]:.4f}")
    print(f"File: PET_TOFNAC_{tag}.nii.gz")
    print(f"PET shape: {PET_data.shape}, PET_max: {np.max(PET_data)}, PET_min: {np.min(PET_data)}")
    print(f"PET mean: {np.mean(PET_data):.4f}, PET std: {np.std(PET_data):.4f}")
    print(f"PET 95th, 99th percentile: {np.percentile(PET_data, 95):.4f} {np.percentile(PET_data, 99):.4f}")
    print(f"PET 99.9th, 99.99th percentile: {np.percentile(PET_data, 99.9):.4f}, {np.percentile(PET_data, 99.99):.4f}")
    dx, dy, dz = PET_file.header.get_zooms()
    print(f"PET physcial spacing: {dx:.4f} {dy:.4f} {dz:.4f}, range: {dx * PET_data.shape[0]:.4f} {dy * PET_data.shape[1]:.4f} {dz * PET_data.shape[2]:.4f}")
    print("--"*25)


# python inspect_B100_data.py
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4055.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1327.4935, CT std: 955.1034
# CT 95th, 99th percentile: 41.0000 120.0000
# CT 99.9th, 99.99th percentile: 918.0000, 1323.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4055.nii.gz
# PET shape: (400, 400, 730), PET_max: 17760.818359375, PET_min: 0.0
# PET mean: 183.6425, PET std: 413.8821
# PET 95th, 99th percentile: 694.0415 1416.4392
# PET 99.9th, 99.99th percentile: 6046.0044, 8998.6143
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4058.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1330.6503, CT std: 952.4782
# CT 95th, 99th percentile: 34.0000 166.0000
# CT 99.9th, 99.99th percentile: 871.0000, 1285.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4058.nii.gz
# PET shape: (400, 400, 730), PET_max: 14123.5205078125, PET_min: 0.0
# PET mean: 202.3224, PET std: 406.8456
# PET 95th, 99th percentile: 787.1746 1483.0740
# PET 99.9th, 99.99th percentile: 5616.6880, 8296.3516
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4061.nii.gz
# CT shape: (467, 467, 730), CT_max: 2556.0, CT_min: -3024.0
# CT mean: -1280.7639, CT std: 988.4540
# CT 95th, 99th percentile: 9.0000 151.0000
# CT 99.9th, 99.99th percentile: 861.0000, 1320.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4061.nii.gz
# PET shape: (400, 400, 730), PET_max: 14852.7275390625, PET_min: 0.0
# PET mean: 205.0741, PET std: 330.8304
# PET 95th, 99th percentile: 722.4671 1455.5408
# PET 99.9th, 99.99th percentile: 3723.6743, 5050.9014
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4066.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1285.9577, CT std: 986.1280
# CT 95th, 99th percentile: 28.0000 130.0000
# CT 99.9th, 99.99th percentile: 834.0000, 1286.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4066.nii.gz
# PET shape: (400, 400, 730), PET_max: 157761.5625, PET_min: 0.0
# PET mean: 225.2975, PET std: 402.0438
# PET 95th, 99th percentile: 755.7207 1455.0632
# PET 99.9th, 99.99th percentile: 5501.7446, 8588.5820
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4068.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1336.9105, CT std: 945.9729
# CT 95th, 99th percentile: 24.0000 95.0000
# CT 99.9th, 99.99th percentile: 885.0000, 1343.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4068.nii.gz
# PET shape: (400, 400, 730), PET_max: 32810.06640625, PET_min: 0.0
# PET mean: 158.4448, PET std: 376.6430
# PET 95th, 99th percentile: 673.9101 1298.6295
# PET 99.9th, 99.99th percentile: 5459.1079, 8664.8713
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4069.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1347.7905, CT std: 939.4597
# CT 95th, 99th percentile: 40.0000 146.0000
# CT 99.9th, 99.99th percentile: 978.0000, 3071.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4069.nii.gz
# PET shape: (400, 400, 730), PET_max: 42701.890625, PET_min: 0.0
# PET mean: 183.2986, PET std: 426.2354
# PET 95th, 99th percentile: 784.7175 1406.9305
# PET 99.9th, 99.99th percentile: 5352.5005, 8478.5674
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4073.nii.gz
# CT shape: (467, 467, 1123), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1373.7047, CT std: 912.5812
# CT 95th, 99th percentile: -35.0000 64.0000
# CT 99.9th, 99.99th percentile: 771.0000, 1320.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1684.5000
# File: PET_TOFNAC_E4073.nii.gz
# PET shape: (400, 400, 1123), PET_max: 11039.7275390625, PET_min: 0.0
# PET mean: 119.4217, PET std: 291.7943
# PET 95th, 99th percentile: 595.5699 1188.0453
# PET 99.9th, 99.99th percentile: 3730.5491, 6459.3540
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1684.5000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4074.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1334.9836, CT std: 945.3486
# CT 95th, 99th percentile: -21.0000 67.0000
# CT 99.9th, 99.99th percentile: 665.0000, 1328.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4074.nii.gz
# PET shape: (400, 400, 652), PET_max: 22264.09765625, PET_min: 0.0
# PET mean: 159.9614, PET std: 313.9261
# PET 95th, 99th percentile: 681.1033 1304.7330
# PET 99.9th, 99.99th percentile: 3141.0166, 6675.3521
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4077.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1316.8949, CT std: 963.0833
# CT 95th, 99th percentile: 27.0000 142.0000
# CT 99.9th, 99.99th percentile: 906.0000, 1293.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4077.nii.gz
# PET shape: (400, 400, 730), PET_max: 49389.99609375, PET_min: 0.0
# PET mean: 225.8623, PET std: 452.2523
# PET 95th, 99th percentile: 810.0273 1465.5720
# PET 99.9th, 99.99th percentile: 6150.0444, 12644.6094
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4078.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1336.6854, CT std: 944.8102
# CT 95th, 99th percentile: 16.0000 79.0000
# CT 99.9th, 99.99th percentile: 637.0000, 1278.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4078.nii.gz
# PET shape: (400, 400, 652), PET_max: 24233.224609375, PET_min: 0.0
# PET mean: 187.3285, PET std: 479.5623
# PET 95th, 99th percentile: 708.2632 1497.5376
# PET 99.9th, 99.99th percentile: 7331.9390, 11366.7305
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4079.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1294.0412, CT std: 980.4676
# CT 95th, 99th percentile: 25.0000 137.0000
# CT 99.9th, 99.99th percentile: 974.0000, 1311.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4079.nii.gz
# PET shape: (400, 400, 730), PET_max: 35323.15625, PET_min: 0.0
# PET mean: 221.0857, PET std: 465.4787
# PET 95th, 99th percentile: 738.1028 1453.6357
# PET 99.9th, 99.99th percentile: 6028.4995, 17016.6387
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4081.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1335.5165, CT std: 947.1934
# CT 95th, 99th percentile: 10.0000 138.0000
# CT 99.9th, 99.99th percentile: 875.0000, 1401.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4081.nii.gz
# PET shape: (400, 400, 730), PET_max: 26956.755859375, PET_min: 0.0
# PET mean: 200.4930, PET std: 424.0349
# PET 95th, 99th percentile: 788.3120 1474.6395
# PET 99.9th, 99.99th percentile: 5845.2305, 9288.4014
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4084.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1360.9675, CT std: 924.2380
# CT 95th, 99th percentile: -3.0000 75.0000
# CT 99.9th, 99.99th percentile: 791.0000, 1311.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4084.nii.gz
# PET shape: (400, 400, 652), PET_max: 31580.765625, PET_min: 0.0
# PET mean: 214.4271, PET std: 508.7779
# PET 95th, 99th percentile: 928.2603 1939.8910
# PET 99.9th, 99.99th percentile: 6819.7021, 11635.5855
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4091.nii.gz
# CT shape: (467, 467, 1123), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1357.3723, CT std: 925.7183
# CT 95th, 99th percentile: -75.0000 53.0000
# CT 99.9th, 99.99th percentile: 673.0000, 1846.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1684.5000
# File: PET_TOFNAC_E4091.nii.gz
# PET shape: (400, 400, 1123), PET_max: 36946.42578125, PET_min: 0.0
# PET mean: 142.2617, PET std: 333.3155
# PET 95th, 99th percentile: 612.3078 1285.4730
# PET 99.9th, 99.99th percentile: 4363.7402, 6547.5825
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1684.5000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4092.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1366.4953, CT std: 921.7532
# CT 95th, 99th percentile: 24.0000 158.0000
# CT 99.9th, 99.99th percentile: 965.0000, 1441.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4092.nii.gz
# PET shape: (400, 400, 730), PET_max: 41925.0859375, PET_min: 0.0
# PET mean: 217.0348, PET std: 471.5064
# PET 95th, 99th percentile: 1004.0350 1788.9098
# PET 99.9th, 99.99th percentile: 4617.1323, 17479.5586
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4094.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1358.0519, CT std: 928.3605
# CT 95th, 99th percentile: 32.0000 112.0000
# CT 99.9th, 99.99th percentile: 793.0000, 1357.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4094.nii.gz
# PET shape: (400, 400, 730), PET_max: 17219.328125, PET_min: 0.0
# PET mean: 182.5201, PET std: 446.2705
# PET 95th, 99th percentile: 758.1809 1629.7223
# PET 99.9th, 99.99th percentile: 5996.2397, 9822.4386
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4096.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1268.8462, CT std: 995.5802
# CT 95th, 99th percentile: 10.0000 73.0000
# CT 99.9th, 99.99th percentile: 595.0000, 1139.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4096.nii.gz
# PET shape: (400, 400, 652), PET_max: 16723.208984375, PET_min: 0.0
# PET mean: 225.9035, PET std: 458.6883
# PET 95th, 99th percentile: 718.4102 1587.6724
# PET 99.9th, 99.99th percentile: 6945.4062, 10329.5879
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4098.nii.gz
# CT shape: (467, 467, 730), CT_max: 2835.0, CT_min: -3024.0
# CT mean: -1288.0871, CT std: 985.1140
# CT 95th, 99th percentile: 31.0000 179.0000
# CT 99.9th, 99.99th percentile: 983.0000, 1327.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4098.nii.gz
# PET shape: (400, 400, 730), PET_max: 85402.5390625, PET_min: 0.0
# PET mean: 236.0231, PET std: 535.5853
# PET 95th, 99th percentile: 757.9211 1585.9081
# PET 99.9th, 99.99th percentile: 7910.6704, 11990.8994
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4099.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1366.8280, CT std: 918.9918
# CT 95th, 99th percentile: -31.0000 77.0000
# CT 99.9th, 99.99th percentile: 823.0000, 1368.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4099.nii.gz
# PET shape: (400, 400, 652), PET_max: 30092.9609375, PET_min: 0.0
# PET mean: 208.0477, PET std: 526.3654
# PET 95th, 99th percentile: 937.7055 1916.9260
# PET 99.9th, 99.99th percentile: 7629.0322, 11036.4375
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4103.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1296.6294, CT std: 978.5117
# CT 95th, 99th percentile: 13.0000 146.0000
# CT 99.9th, 99.99th percentile: 949.0000, 1361.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4103.nii.gz
# PET shape: (400, 400, 730), PET_max: 17685.927734375, PET_min: 0.0
# PET mean: 204.9737, PET std: 399.1530
# PET 95th, 99th percentile: 684.6153 1435.5078
# PET 99.9th, 99.99th percentile: 5714.8027, 8417.4697
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4105.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1367.5618, CT std: 918.0734
# CT 95th, 99th percentile: -44.0000 73.0000
# CT 99.9th, 99.99th percentile: 802.0000, 1370.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4105.nii.gz
# PET shape: (400, 400, 652), PET_max: 143189.59375, PET_min: 0.0
# PET mean: 234.7533, PET std: 673.3630
# PET 95th, 99th percentile: 952.2356 2145.8472
# PET 99.9th, 99.99th percentile: 8877.7354, 14343.9018
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4106.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1349.2388, CT std: 934.3786
# CT 95th, 99th percentile: -22.0000 77.0000
# CT 99.9th, 99.99th percentile: 928.0000, 1393.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4106.nii.gz
# PET shape: (400, 400, 652), PET_max: 22079.6484375, PET_min: 0.0
# PET mean: 174.0280, PET std: 405.3431
# PET 95th, 99th percentile: 732.3031 1398.4349
# PET 99.9th, 99.99th percentile: 6045.1836, 9098.8848
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4114.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1302.8876, CT std: 970.4794
# CT 95th, 99th percentile: -16.0000 151.0000
# CT 99.9th, 99.99th percentile: 775.0000, 1274.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4114.nii.gz
# PET shape: (400, 400, 652), PET_max: 21689.37890625, PET_min: 0.0
# PET mean: 203.1263, PET std: 490.9974
# PET 95th, 99th percentile: 706.5093 1569.0918
# PET 99.9th, 99.99th percentile: 7786.0581, 11092.7012
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4115.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1317.8033, CT std: 959.5238
# CT 95th, 99th percentile: -1.0000 91.0000
# CT 99.9th, 99.99th percentile: 617.0000, 1275.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4115.nii.gz
# PET shape: (400, 400, 652), PET_max: 16663.388671875, PET_min: 0.0
# PET mean: 265.1748, PET std: 439.3750
# PET 95th, 99th percentile: 942.9610 1722.9247
# PET 99.9th, 99.99th percentile: 5697.6162, 8244.0029
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4118.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1339.2328, CT std: 943.9874
# CT 95th, 99th percentile: 0.0000 138.0000
# CT 99.9th, 99.99th percentile: 904.0000, 1353.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4118.nii.gz
# PET shape: (400, 400, 652), PET_max: 19509.57421875, PET_min: 0.0
# PET mean: 207.1719, PET std: 394.9799
# PET 95th, 99th percentile: 869.9614 1585.8772
# PET 99.9th, 99.99th percentile: 4760.9082, 7892.2544
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4120.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1371.0976, CT std: 916.2626
# CT 95th, 99th percentile: -22.0000 88.0000
# CT 99.9th, 99.99th percentile: 978.0000, 1402.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4120.nii.gz
# PET shape: (400, 400, 652), PET_max: 49883.9375, PET_min: 0.0
# PET mean: 236.7557, PET std: 669.1231
# PET 95th, 99th percentile: 1032.2672 2030.6541
# PET 99.9th, 99.99th percentile: 8623.0156, 25426.0645
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4124.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1334.7058, CT std: 948.4077
# CT 95th, 99th percentile: 27.0000 114.0000
# CT 99.9th, 99.99th percentile: 925.0000, 1364.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4124.nii.gz
# PET shape: (400, 400, 730), PET_max: 53763.7578125, PET_min: 0.0
# PET mean: 165.8955, PET std: 394.0527
# PET 95th, 99th percentile: 668.9820 1259.8256
# PET 99.9th, 99.99th percentile: 4871.7686, 7663.9639
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4125.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1305.1136, CT std: 972.5001
# CT 95th, 99th percentile: 39.0000 136.0000
# CT 99.9th, 99.99th percentile: 848.0000, 1242.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4125.nii.gz
# PET shape: (400, 400, 730), PET_max: 48412.93359375, PET_min: 0.0
# PET mean: 219.7545, PET std: 488.7322
# PET 95th, 99th percentile: 766.1976 1557.6443
# PET 99.9th, 99.99th percentile: 6774.7163, 10544.7656
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4128.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1266.9371, CT std: 1000.3400
# CT 95th, 99th percentile: 24.0000 156.0000
# CT 99.9th, 99.99th percentile: 890.0000, 3071.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4128.nii.gz
# PET shape: (400, 400, 730), PET_max: 13001.5830078125, PET_min: 0.0
# PET mean: 216.7243, PET std: 346.7763
# PET 95th, 99th percentile: 692.3644 1340.2572
# PET 99.9th, 99.99th percentile: 4557.1148, 6858.5366
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4129.nii.gz
# CT shape: (467, 467, 1201), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1383.7153, CT std: 901.6189
# CT 95th, 99th percentile: -95.0000 56.0000
# CT 99.9th, 99.99th percentile: 578.0000, 1257.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1801.5000
# File: PET_TOFNAC_E4129.nii.gz
# PET shape: (400, 400, 1201), PET_max: 21370.171875, PET_min: 0.0
# PET mean: 129.1957, PET std: 413.8865
# PET 95th, 99th percentile: 587.8177 1447.8451
# PET 99.9th, 99.99th percentile: 6275.2642, 10870.1738
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1801.5000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4130.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1341.3339, CT std: 942.3211
# CT 95th, 99th percentile: 19.0000 98.0000
# CT 99.9th, 99.99th percentile: 884.0000, 1365.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4130.nii.gz
# PET shape: (400, 400, 730), PET_max: 16724.40234375, PET_min: 0.0
# PET mean: 145.8345, PET std: 347.2950
# PET 95th, 99th percentile: 610.6750 1215.6993
# PET 99.9th, 99.99th percentile: 4711.5332, 10398.8721
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4131.nii.gz
# CT shape: (467, 467, 652), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1332.8596, CT std: 947.9017
# CT 95th, 99th percentile: -10.0000 96.0000
# CT 99.9th, 99.99th percentile: 876.0000, 1371.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 978.0000
# File: PET_TOFNAC_E4131.nii.gz
# PET shape: (400, 400, 652), PET_max: 20839.615234375, PET_min: 0.0
# PET mean: 149.1764, PET std: 427.3092
# PET 95th, 99th percentile: 571.0865 1219.3622
# PET 99.9th, 99.99th percentile: 7062.9570, 10764.9316
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 978.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4134.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1354.9145, CT std: 930.2267
# CT 95th, 99th percentile: 7.0000 97.0000
# CT 99.9th, 99.99th percentile: 886.0000, 1429.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4134.nii.gz
# PET shape: (400, 400, 730), PET_max: 17405.794921875, PET_min: 0.0
# PET mean: 183.5796, PET std: 353.1797
# PET 95th, 99th percentile: 817.7519 1519.1539
# PET 99.9th, 99.99th percentile: 3915.9985, 5707.3130
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4137.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1312.4505, CT std: 964.3584
# CT 95th, 99th percentile: 7.0000 94.0000
# CT 99.9th, 99.99th percentile: 851.0000, 1269.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4137.nii.gz
# PET shape: (400, 400, 730), PET_max: 17716.029296875, PET_min: 0.0
# PET mean: 224.3810, PET std: 411.2473
# PET 95th, 99th percentile: 862.4888 1628.8790
# PET 99.9th, 99.99th percentile: 5518.1206, 8175.5303
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4138.nii.gz
# CT shape: (467, 467, 730), CT_max: 2448.0, CT_min: -3024.0
# CT mean: -1351.9604, CT std: 933.0893
# CT 95th, 99th percentile: 17.0000 89.0000
# CT 99.9th, 99.99th percentile: 887.0000, 1337.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4138.nii.gz
# PET shape: (400, 400, 730), PET_max: 945832.5625, PET_min: 0.0
# PET mean: 195.1416, PET std: 1016.7767
# PET 95th, 99th percentile: 796.4868 1659.8260
# PET 99.9th, 99.99th percentile: 7301.1807, 10833.3418
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4139.nii.gz
# CT shape: (467, 467, 730), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1334.2335, CT std: 947.3138
# CT 95th, 99th percentile: 6.0000 150.0000
# CT 99.9th, 99.99th percentile: 880.0000, 1430.0000
# CT physcial spacing: 1.5000 1.5000 1.5000, range: 700.5000 700.5000 1095.0000
# File: PET_TOFNAC_E4139.nii.gz
# PET shape: (400, 400, 730), PET_max: 19664.7734375, PET_min: 0.0
# PET mean: 176.7051, PET std: 340.5328
# PET 95th, 99th percentile: 736.8500 1362.2250
# PET 99.9th, 99.99th percentile: 3969.0986, 7113.1736
# PET physcial spacing: 1.5000 1.5000 1.5000, range: 600.0000 600.0000 1095.0000
# --------------------------------------------------
# root@8d6d1514c91c:/Ammongus#
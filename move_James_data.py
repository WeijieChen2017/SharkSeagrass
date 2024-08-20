# 'PET TOFNAC E4055 B100'  'PET TOFNAC E4084 B100'  'PET TOFNAC E4115 B100'
# 'PET TOFNAC E4058 B100'  'PET TOFNAC E4087 B100'  'PET TOFNAC E4118 B100'
# 'PET TOFNAC E4061 B100'  'PET TOFNAC E4091 B100'  'PET TOFNAC E4120 B100'
# 'PET TOFNAC E4063 B100'  'PET TOFNAC E4092 B100'  'PET TOFNAC E4124 B100'
# 'PET TOFNAC E4066 B100'  'PET TOFNAC E4094 B100'  'PET TOFNAC E4125 B100'
# 'PET TOFNAC E4068 B100'  'PET TOFNAC E4096 B100'  'PET TOFNAC E4128 B100'
# 'PET TOFNAC E4069 B100'  'PET TOFNAC E4097 B100'  'PET TOFNAC E4129 B100'
# 'PET TOFNAC E4073 B100'  'PET TOFNAC E4098 B100'  'PET TOFNAC E4130 B100'
# 'PET TOFNAC E4074 B100'  'PET TOFNAC E4099 B100'  'PET TOFNAC E4131 B100'
# 'PET TOFNAC E4077 B100'  'PET TOFNAC E4102 B100'  'PET TOFNAC E4134 B100'
# 'PET TOFNAC E4078 B100'  'PET TOFNAC E4103 B100'  'PET TOFNAC E4137 B100'
# 'PET TOFNAC E4079 B100'  'PET TOFNAC E4105 B100'  'PET TOFNAC E4138 B100'
# 'PET TOFNAC E4080 B100'  'PET TOFNAC E4106 B100'  'PET TOFNAC E4139 B100'
# 'PET TOFNAC E4081 B100'  'PET TOFNAC E4114 B100'

tag_list = [
    "E4055", "E4058", "E4061", "E4063", "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079", "E4080", "E4081", "E4084",
    "E4087", "E4091", "E4092", "E4094", "E4096",
    "E4097", "E4098", "E4099", "E4102", "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
]
# import os
# for tag in tag_list:
#     cmd = f"cp CTACIVV_{tag[1:]}.nii /shares/mimrtl/Users/Winston/files_to_dgx/SharkSeagrass/"
#     print(cmd)

# # gzip all files
# cmd = "gzip "
# for tag in tag_list:
#     cmd += f"CTACIVV_{tag[1:]}.nii "
# print(cmd)

# # output will be
# cmd = "gzip CTACIVV_4055.nii CTACIVV_4058.nii CTACIVV_4061.nii CTACIVV_4063.nii CTACIVV_4066.nii CTACIVV_4068.nii CTACIVV_4069.nii CTACIVV_4073.nii CTACIVV_4074.nii CTACIVV_4077.nii CTACIVV_4078.nii CTACIVV_4079.nii CTACIVV_4080.nii CTACIVV_4081.nii CTACIVV_4084.nii CTACIVV_4087.nii CTACIVV_4091.nii CTACIVV_4092.nii CTACIVV_4094.nii CTACIVV_4096.nii CTACIVV_4097.nii CTACIVV_4098.nii CTACIVV_4099.nii CTACIVV_4102.nii CTACIVV_4103.nii CTACIVV_4105.nii CTACIVV_4106.nii CTACIVV_4114.nii CTACIVV_4115.nii CTACIVV_4118.nii CTACIVV_4120.nii CTACIVV_4124.nii CTACIVV_4125.nii CTACIVV_4128.nii CTACIVV_4129.nii CTACIVV_4130.nii CTACIVV_4131.nii CTACIVV_4134.nii CTACIVV_4137.nii CTACIVV_4138.nii CTACIVV_4139.nii"

#  PET_TOFNAC_E4055_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421085741_5.nii
#  PET_TOFNAC_E4058_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170204124826_5.nii
#  PET_TOFNAC_E4061_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170121114202_5.nii
# 'PET_TOFNAC_E4063_B100_5.8_PET_CT_(1.8m)_WB_HEAD_TO_FEET_(FEET_IN,_5'\''11_--6'\''3_)_20170130085046_5.nii'
#  PET_TOFNAC_E4066_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170422103620_5.nii
#  PET_TOFNAC_E4068_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170217115607_5.nii
#  PET_TOFNAC_E4069_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170128114753_5.nii
# 'PET_TOFNAC_E4073_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170204100835_5.nii'
#  PET_TOFNAC_E4074_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170302113826_5.nii
#  PET_TOFNAC_E4077_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170210105129_5.nii
#  PET_TOFNAC_E4078_B100_5.1_PET_CT_HEAD_TO_PELVIS_20171116113128_5.nii
#  PET_TOFNAC_E4079_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170212090413_5.nii
# 'PET_TOFNAC_E4080_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170204090439_5.nii'
#  PET_TOFNAC_E4081_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170323113257_5.nii
#  PET_TOFNAC_E4084_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170324085841_5.nii
# 'PET_TOFNAC_E4087_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170519095346_5.nii'
# 'PET_TOFNAC_E4091_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170309134559_5.nii'
#  PET_TOFNAC_E4092_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170310104511_5.nii
#  PET_TOFNAC_E4094_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170213135043_5.nii
#  PET_TOFNAC_E4096_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170429085017_5.nii
# 'PET_TOFNAC_E4097_B100_5.9_PET_CT_(2m)_WB_FEET_TO_HEAD_20170304115940_5.nii'
#  PET_TOFNAC_E4098_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170302133037_5.nii
#  PET_TOFNAC_E4099_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170327105032_5.nii
# 'PET_TOFNAC_E4102_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170819121826_5.nii'
#  PET_TOFNAC_E4103_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170220132322_5.nii
#  PET_TOFNAC_E4105_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170227085717_5.nii
#  PET_TOFNAC_E4106_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170402110347_5.nii
#  PET_TOFNAC_E4114_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170306105211_5.nii
#  PET_TOFNAC_E4115_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421104102_5.nii
#  PET_TOFNAC_E4118_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421112735_5.nii
#  PET_TOFNAC_E4120_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170312102156_5.nii
#  PET_TOFNAC_E4124_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421131143_5.nii
#  PET_TOFNAC_E4125_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170311135526_5.nii
#  PET_TOFNAC_E4128_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170317105547_5.nii
# 'PET_TOFNAC_E4129_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170617085520_5.nii'
#  PET_TOFNAC_E4130_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170401101057_5.nii
#  PET_TOFNAC_E4131_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170413110841_5.nii
#  PET_TOFNAC_E4134_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170414104935_5.nii
#  PET_TOFNAC_E4137_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170325084653_5.nii

# mv PET_TOFNAC_E4055_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421085741_5.nii PET_TOFNAC_E4055.nii
# mv PET_TOFNAC_E4058_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170204124826_5.nii PET_TOFNAC_E4058.nii
# mv PET_TOFNAC_E4061_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170121114202_5.nii PET_TOFNAC_E4061.nii
# mv 'PET_TOFNAC_E4063_B100_5.8_PET_CT_(1.8m)_WB_HEAD_TO_FEET_(FEET_IN,_5'\''11_--6'\''3_)_20170130085046_5.nii' PET_TOFNAC_E4063.nii
# mv PET_TOFNAC_E4066_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170422103620_5.nii PET_TOFNAC_E4066.nii
# mv PET_TOFNAC_E4068_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170217115607_5.nii PET_TOFNAC_E4068.nii
# mv PET_TOFNAC_E4069_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170128114753_5.nii PET_TOFNAC_E4069.nii
# mv 'PET_TOFNAC_E4073_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170204100835_5.nii' PET_TOFNAC_E4073.nii
# mv PET_TOFNAC_E4074_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170302113826_5.nii PET_TOFNAC_E4074.nii
# mv PET_TOFNAC_E4077_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170210105129_5.nii PET_TOFNAC_E4077.nii
# mv PET_TOFNAC_E4078_B100_5.1_PET_CT_HEAD_TO_PELVIS_20171116113128_5.nii PET_TOFNAC_E4078.nii
# mv PET_TOFNAC_E4079_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170212090413_5.nii PET_TOFNAC_E4079.nii
# mv 'PET_TOFNAC_E4080_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170204090439_5.nii' PET_TOFNAC_E4080.nii
# mv PET_TOFNAC_E4081_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170323113257_5.nii PET_TOFNAC_E4081.nii
# mv PET_TOFNAC_E4084_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170324085841_5.nii PET_TOFNAC_E4084.nii
# mv 'PET_TOFNAC_E4087_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170519095346_5.nii' PET_TOFNAC_E4087.nii
# mv 'PET_TOFNAC_E4091_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170309134559_5.nii' PET_TOFNAC_E4091.nii
# mv PET_TOFNAC_E4092_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170310104511_5.nii PET_TOFNAC_E4092.nii
# mv PET_TOFNAC_E4094_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170213135043_5.nii PET_TOFNAC_E4094.nii
# mv PET_TOFNAC_E4096_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170429085017_5.nii PET_TOFNAC_E4096.nii
# mv 'PET_TOFNAC_E4097_B100_5.9_PET_CT_(2m)_WB_FEET_TO_HEAD_20170304115940_5.nii' PET_TOFNAC_E4097.nii
# mv PET_TOFNAC_E4098_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170302133037_5.nii PET_TOFNAC_E4098.nii
# mv PET_TOFNAC_E4099_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170327105032_5.nii PET_TOFNAC_E4099.nii
# mv 'PET_TOFNAC_E4102_B100_5.7_PET_CT_(2m)_WB_HEAD_TO_FEET_(FEET_IN,_6'\''3_+++)_20170819121826_5.nii' PET_TOFNAC_E4102.nii
# mv PET_TOFNAC_E4103_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170220132322_5.nii PET_TOFNAC_E4103.nii
# mv PET_TOFNAC_E4105_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170227085717_5.nii PET_TOFNAC_E4105.nii
# mv PET_TOFNAC_E4106_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170402110347_5.nii PET_TOFNAC_E4106.nii
# mv PET_TOFNAC_E4114_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170306105211_5.nii PET_TOFNAC_E4114.nii
# mv PET_TOFNAC_E4115_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421104102_5.nii PET_TOFNAC_E4115.nii
# mv PET_TOFNAC_E4118_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421112735_5.nii PET_TOFNAC_E4118.nii
# mv PET_TOFNAC_E4120_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170312102156_5.nii PET_TOFNAC_E4120.nii
# mv PET_TOFNAC_E4124_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170421131143_5.nii PET_TOFNAC_E4124.nii
# mv PET_TOFNAC_E4125_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170311135526_5.nii PET_TOFNAC_E4125.nii
# mv PET_TOFNAC_E4128_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170317105547_5.nii PET_TOFNAC_E4128.nii
# mv 'PET_TOFNAC_E4129_B100_5.4_PET_CT_WB_REG_TABLE_HEAD_TO_FEET_(HEAD_IN,_UP_TO_5'\''4_)_20170617085520_5.nii' PET_TOFNAC_E4129.nii
# mv PET_TOFNAC_E4130_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170401101057_5.nii PET_TOFNAC_E4130.nii
# mv PET_TOFNAC_E4131_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170413110841_5.nii PET_TOFNAC_E4131.nii
# mv PET_TOFNAC_E4134_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170414104935_5.nii PET_TOFNAC_E4134.nii
# mv PET_TOFNAC_E4137_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170325084653_5.nii PET_TOFNAC_E4137.nii
# mv PET_TOFNAC_E4138_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170325093015_5.nii PET_TOFNAC_E4138.nii
# mv PET_TOFNAC_E4139_B100_5.1_PET_CT_HEAD_TO_PELVIS_20170427104545_5.nii PET_TOFNAC_E4139.nii

# gzip cmd:
# gzip PET_TOFNAC_E4055.nii PET_TOFNAC_E4058.nii PET_TOFNAC_E4061.nii PET_TOFNAC_E4063.nii PET_TOFNAC_E4066.nii PET_TOFNAC_E4068.nii PET_TOFNAC_E4069.nii PET_TOFNAC_E4073.nii PET_TOFNAC_E4074.nii PET_TOFNAC_E4077.nii PET_TOFNAC_E4078.nii PET_TOFNAC_E4079.nii PET_TOFNAC_E4080.nii PET_TOFNAC_E4081.nii PET_TOFNAC_E4084.nii PET_TOFNAC_E4087.nii PET_TOFNAC_E4091.nii PET_TOFNAC_E4092.nii PET_TOFNAC_E4094.nii PET_TOFNAC_E4096.nii PET_TOFNAC_E4097.nii PET_TOFNAC_E4098.nii PET_TOFNAC_E4099.nii PET_TOFNAC_E4102.nii PET_TOFNAC_E4103.nii PET_TOFNAC_E4105.nii PET_TOFNAC_E4106.nii PET_TOFNAC_E4114.nii PET_TOFNAC_E4115.nii PET_TOFNAC_E4118.nii PET_TOFNAC_E4120.nii PET_TOFNAC_E4124.nii PET_TOFNAC_E4125.nii PET_TOFNAC_E4128.nii PET_TOFNAC_E4129.nii PET_TOFNAC_E4130.nii PET_TOFNAC_E4131.nii PET_TOFNAC_E4134.nii PET_TOFNAC_E4137.nii PET_TOFNAC_E4138.nii PET_TOFNAC_E4139.nii



# here we load the data
import os
import nibabel as nib
import numpy as np

for tag in tag_list:
    CT_path = f"./B100/CTACIVV/CTACIVV_{tag[1:]}.nii.gz"
    PET_path = f"./B100/TOFNAC/PET_TOFNAC_{tag}.nii.gz"
    CT_file = nib.load(CT_path)
    PET_file = nib.load(PET_path)
    CT_data = CT_file.get_fdata()
    PET_data = PET_file.get_fdata()

    print("<"*50)
    print(f"File: CTACIVV_{tag[1:]}.nii.gz")
    print(f"CT shape: {CT_data.shape}, CT_max: {np.max(CT_data)}, CT_min: {np.min(CT_data)}")
    print(f"CT mean: {np.mean(CT_data)}, CT std: {np.std(CT_data)}")
    print(f"CT 95th percentile: {np.percentile(CT_data, 95)}")
    print(f"CT 99th percentile: {np.percentile(CT_data, 99)}")
    print(f"CT 99.9th percentile: {np.percentile(CT_data, 99.9)}")
    print(f"CT 99.99th percentile: {np.percentile(CT_data, 99.99)}")
    print(f"CT physcial spacing: {CT_file.header.get_zooms()}")
    print(f"CT physical range: {CT_file.header.get_zooms() * np.array(CT_data.shape)}")
    print(">"*50)
    print(f"File: PET_TOFNAC_{tag}.nii.gz")
    print(f"PET shape: {PET_data.shape}, PET_max: {np.max(PET_data)}, PET_min: {np.min(PET_data)}")
    print(f"PET mean: {np.mean(PET_data)}, PET std: {np.std(PET_data)}")
    print(f"PET 95th percentile: {np.percentile(PET_data, 95)}")
    print(f"PET 99th percentile: {np.percentile(PET_data, 99)}")
    print(f"PET 99.9th percentile: {np.percentile(PET_data, 99.9)}")
    print(f"PET 99.99th percentile: {np.percentile(PET_data, 99.99)}")
    print(f"PET physcial spacing: {PET_file.header.get_zooms()}")
    print(f"PET physical range: {PET_file.header.get_zooms() * np.array(PET_data.shape)}")

    print("<--->")

# root@fac5e29efbe7:/SharkSeagrass# python move_James_data.py 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4055.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1325.0898544994752, CT std: 953.6541292176028
# CT 95th percentile: 41.0
# CT 99th percentile: 120.0
# CT 99.9th percentile: 920.0
# CT 99.99th percentile: 1323.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4055.nii.gz
# PET shape: (256, 256, 335), PET_max: 17760.818359375, PET_min: 0.0
# PET mean: 183.62721659836285, PET std: 413.50609920602
# PET 95th percentile: 694.0185546875
# PET 99th percentile: 1416.9344824218751
# PET 99.9th percentile: 6037.226290527875
# PET 99.99th percentile: 8998.540369042927
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4058.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1328.263851712355, CT std: 951.0194040651713
# CT 95th percentile: 35.0
# CT 99th percentile: 167.0
# CT 99.9th percentile: 872.0
# CT 99.99th percentile: 1287.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4058.nii.gz
# PET shape: (256, 256, 335), PET_max: 14123.5205078125, PET_min: 0.0
# PET mean: 202.37799785421308, PET std: 406.91600721850847
# PET 95th percentile: 787.3607177734375
# PET 99th percentile: 1483.7096557617197
# PET 99.9th percentile: 5627.462130371625
# PET 99.99th percentile: 8278.887145897963
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4061.nii.gz
# CT shape: (512, 512, 335), CT_max: 2556.0, CT_min: -3024.0
# CT mean: -1278.4055572623638, CT std: 986.9157033829148
# CT 95th percentile: 9.0
# CT 99th percentile: 152.0
# CT 99.9th percentile: 862.0
# CT 99.99th percentile: 1322.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4061.nii.gz
# PET shape: (256, 256, 335), PET_max: 14852.7275390625, PET_min: 0.0
# PET mean: 205.16697489679817, PET std: 330.8777203091076
# PET 95th percentile: 722.6995239257812
# PET 99th percentile: 1455.867064208985
# PET 99.9th percentile: 3723.3442490235066
# PET 99.99th percentile: 5044.801534619064
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4063.nii.gz
# CT shape: (512, 512, 582), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1330.6247499538042, CT std: 946.0877872599821
# CT 95th percentile: -5.0
# CT 99th percentile: 133.0
# CT 99.9th percentile: 839.0
# CT 99.99th percentile: 1402.0
# CT physcial spacing: (1.367188, 1.367188, 3.2699966)
# CT physical range: [ 700.00024414  700.00024414 1903.13804626]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4063.nii.gz
# PET shape: (256, 256, 587), PET_max: 15607.74609375, PET_min: 0.0
# PET mean: 115.00201623273169, PET std: 310.65738310917015
# PET 95th percentile: 481.2243347167969
# PET 99th percentile: 1177.6118127441368
# PET 99.9th percentile: 4402.09033203125
# PET 99.99th percentile: 7259.262572753782
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.         600.        1919.4899888]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4066.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1283.5064851106104, CT std: 984.6580634268515
# CT 95th percentile: 28.0
# CT 99th percentile: 131.0
# CT 99.9th percentile: 836.0
# CT 99.99th percentile: 1286.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4066.nii.gz
# PET shape: (256, 256, 335), PET_max: 157761.5625, PET_min: 0.0
# PET mean: 225.31884874954852, PET std: 401.14162432703256
# PET 95th percentile: 755.7089324951173
# PET 99th percentile: 1454.3264929199227
# PET 99.9th percentile: 5493.66990429721
# PET 99.99th percentile: 8591.171784179089
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4068.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1334.5019233817484, CT std: 944.5365499812729
# CT 95th percentile: 24.0
# CT 99th percentile: 96.0
# CT 99.9th percentile: 887.0
# CT 99.99th percentile: 1345.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4068.nii.gz
# PET shape: (256, 256, 335), PET_max: 32810.06640625, PET_min: 0.0
# PET mean: 158.45573215913845, PET std: 376.6125140782524
# PET 95th percentile: 673.7910949707034
# PET 99th percentile: 1298.747141113282
# PET 99.9th percentile: 5460.641814941993
# PET 99.99th percentile: 8663.285452241678
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4069.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1345.4033560567827, CT std: 938.0239198077746
# CT 95th percentile: 40.0
# CT 99th percentile: 146.0
# CT 99.9th percentile: 979.0
# CT 99.99th percentile: 3071.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4069.nii.gz
# PET shape: (256, 256, 335), PET_max: 42701.890625, PET_min: 0.0
# PET mean: 183.33474824515588, PET std: 426.7148340929178
# PET 95th percentile: 784.6783447265625
# PET 99th percentile: 1406.6301171875002
# PET 99.9th percentile: 5342.015049316704
# PET 99.99th percentile: 8435.23987294891
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4073.nii.gz
# CT shape: (512, 512, 515), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1371.3630399315102, CT std: 911.1040487826026
# CT 95th percentile: -35.0
# CT 99th percentile: 64.0
# CT 99.9th percentile: 768.0
# CT 99.99th percentile: 1316.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1684.04999018]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4073.nii.gz
# PET shape: (256, 256, 515), PET_max: 11039.7275390625, PET_min: 0.0
# PET mean: 119.47040581250363, PET std: 291.9589832891224
# PET 95th percentile: 595.784259033202
# PET 99th percentile: 1188.6036376953125
# PET 99.9th percentile: 3730.52824536148
# PET 99.99th percentile: 6467.3383021483605
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1684.04999018]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4074.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1332.5783217567266, CT std: 943.8919074382761
# CT 95th percentile: -21.0
# CT 99th percentile: 67.0
# CT 99.9th percentile: 665.0
# CT 99.99th percentile: 1330.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4074.nii.gz
# PET shape: (256, 256, 299), PET_max: 22264.09765625, PET_min: 0.0
# PET mean: 160.04427331703621, PET std: 315.32924588711825
# PET 95th percentile: 681.0687225341787
# PET 99th percentile: 1304.5428466796875
# PET 99.9th percentile: 3141.338440674055
# PET 99.99th percentile: 7128.356581932621
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4077.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1314.5201045819183, CT std: 961.575813031653
# CT 95th percentile: 27.0
# CT 99th percentile: 142.0
# CT 99.9th percentile: 904.0
# CT 99.99th percentile: 1293.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4077.nii.gz
# PET shape: (256, 256, 335), PET_max: 49389.99609375, PET_min: 0.0
# PET mean: 225.9739548199282, PET std: 451.92619976079345
# PET 95th percentile: 810.4542236328125
# PET 99th percentile: 1466.6522778320314
# PET 99.9th percentile: 6146.843919921921
# PET 99.99th percentile: 12620.149173045429
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4078.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1334.2484119121846, CT std: 943.3892392816143
# CT 95th percentile: 16.0
# CT 99th percentile: 79.0
# CT 99.9th percentile: 637.0
# CT 99.99th percentile: 1280.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4078.nii.gz
# PET shape: (256, 256, 299), PET_max: 24233.224609375, PET_min: 0.0
# PET mean: 187.4102811571743, PET std: 479.5077809166054
# PET 95th percentile: 708.3216552734375
# PET 99th percentile: 1498.3742431640676
# PET 99.9th percentile: 7327.491104494358
# PET 99.99th percentile: 11357.584366208299
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4079.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1291.5940951674731, CT std: 979.0112496220056
# CT 95th percentile: 25.0
# CT 99th percentile: 138.0
# CT 99.9th percentile: 976.0
# CT 99.99th percentile: 1311.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4079.nii.gz
# PET shape: (256, 256, 335), PET_max: 35323.15625, PET_min: 0.0
# PET mean: 221.16831246109965, PET std: 465.73862993432397
# PET 95th percentile: 738.3837890625
# PET 99th percentile: 1453.6542456054694
# PET 99.9th percentile: 6023.733990234592
# PET 99.99th percentile: 16962.04547361141
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4080.nii.gz
# CT shape: (512, 512, 606), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1344.948754798461, CT std: 934.5113161927331
# CT 95th percentile: 8.0
# CT 99th percentile: 121.0
# CT 99.9th percentile: 758.0
# CT 99.99th percentile: 1282.0
# CT physcial spacing: (1.367188, 1.367188, 5.0)
# CT physical range: [ 700.00024414  700.00024414 3030.        ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4080.nii.gz
# PET shape: (256, 256, 623), PET_max: 18070.34375, PET_min: 0.0
# PET mean: 157.95579485796773, PET std: 374.5476156482859
# PET 95th percentile: 700.1454620361324
# PET 99th percentile: 1536.04541015625
# PET 99.9th percentile: 4792.3849296876
# PET 99.99th percentile: 8592.034636621058
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         2037.20998812]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4081.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1333.162839826897, CT std: 945.7057423215084
# CT 95th percentile: 10.0
# CT 99th percentile: 139.0
# CT 99.9th percentile: 877.0
# CT 99.99th percentile: 1405.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4081.nii.gz
# PET shape: (256, 256, 335), PET_max: 26956.755859375, PET_min: 0.0
# PET mean: 200.47564655278128, PET std: 423.6972188411925
# PET 95th percentile: 788.1445922851562
# PET 99th percentile: 1474.2020263671875
# PET 99.9th percentile: 5830.9957573243355
# PET 99.99th percentile: 9280.550371770853
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4084.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1358.6491009766441, CT std: 922.7393304657074
# CT 95th percentile: -3.0
# CT 99th percentile: 75.0
# CT 99.9th percentile: 791.0
# CT 99.99th percentile: 1313.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4084.nii.gz
# PET shape: (256, 256, 299), PET_max: 31580.765625, PET_min: 0.0
# PET mean: 214.46172504080917, PET std: 508.80973104770845
# PET 95th percentile: 927.567501831051
# PET 99th percentile: 1940.592747802743
# PET 99.9th percentile: 6820.75341796875
# PET 99.99th percentile: 11632.305837393902
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4087.nii.gz
# CT shape: (512, 512, 582), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1341.759744231234, CT std: 936.3938912434564
# CT 95th percentile: -34.0
# CT 99th percentile: 82.0
# CT 99.9th percentile: 817.0
# CT 99.99th percentile: 1337.0
# CT physcial spacing: (1.367188, 1.367188, 5.0)
# CT physical range: [ 700.00024414  700.00024414 2910.        ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4087.nii.gz
# PET shape: (256, 256, 587), PET_max: 19128.138671875, PET_min: 0.0
# PET mean: 151.53543743321237, PET std: 366.5669833877113
# PET 95th percentile: 629.19580078125
# PET 99th percentile: 1341.7894128417902
# PET 99.9th percentile: 5236.203984375054
# PET 99.99th percentile: 9161.257430174635
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.         600.        1919.4899888]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4091.nii.gz
# CT shape: (512, 512, 515), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1355.0166310652946, CT std: 924.2388811074119
# CT 95th percentile: -75.0
# CT 99th percentile: 53.0
# CT 99.9th percentile: 673.0
# CT 99.99th percentile: 1847.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1684.04999018]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4091.nii.gz
# PET shape: (256, 256, 515), PET_max: 36946.42578125, PET_min: 0.0
# PET mean: 142.28343552391738, PET std: 332.9720451218871
# PET 95th percentile: 612.3577270507812
# PET 99th percentile: 1285.9175793457016
# PET 99.9th percentile: 4359.567023437819
# PET 99.99th percentile: 6538.3594805173125
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1684.04999018]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4092.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1364.1443947179994, CT std: 920.3046862323015
# CT 95th percentile: 24.0
# CT 99th percentile: 159.0
# CT 99.9th percentile: 968.0
# CT 99.99th percentile: 1441.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4092.nii.gz
# PET shape: (256, 256, 335), PET_max: 41925.0859375, PET_min: 0.0
# PET mean: 217.01492202411438, PET std: 470.67226724366617
# PET 95th percentile: 1003.8654693603523
# PET 99th percentile: 1788.9376184082034
# PET 99.9th percentile: 4612.226323242368
# PET 99.99th percentile: 17364.094003511287
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4094.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1355.7172800092555, CT std: 926.8981787638952
# CT 95th percentile: 32.0
# CT 99th percentile: 112.0
# CT 99.9th percentile: 797.0
# CT 99.99th percentile: 1359.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4094.nii.gz
# PET shape: (256, 256, 335), PET_max: 17219.328125, PET_min: 0.0
# PET mean: 182.52056549283475, PET std: 446.1944122457029
# PET 95th percentile: 757.8424224853518
# PET 99th percentile: 1629.4374572753918
# PET 99.9th percentile: 5995.294273926362
# PET 99.99th percentile: 9830.24544365185
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4096.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1266.3222111985835, CT std: 994.1269233759775
# CT 95th percentile: 10.0
# CT 99th percentile: 73.0
# CT 99.9th percentile: 596.0
# CT 99.99th percentile: 1140.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4096.nii.gz
# PET shape: (256, 256, 299), PET_max: 16723.208984375, PET_min: 0.0
# PET mean: 225.98472626944712, PET std: 458.79442628014755
# PET 95th percentile: 718.5300903320312
# PET 99th percentile: 1586.8527197265757
# PET 99.9th percentile: 6944.041947265767
# PET 99.99th percentile: 10340.031302538926
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4097.nii.gz
# CT shape: (512, 512, 553), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1356.9379748780732, CT std: 924.2524032268626
# CT 95th percentile: -47.0
# CT 99th percentile: 82.0
# CT 99.9th percentile: 913.0
# CT 99.99th percentile: 1348.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1808.30998945]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4097.nii.gz
# PET shape: (256, 256, 551), PET_max: 22466.40625, PET_min: 0.0
# PET mean: 146.446926918152, PET std: 413.57742126992423
# PET 95th percentile: 638.8474884033203
# PET 99th percentile: 1460.2502990722596
# PET 99.9th percentile: 6145.896259765868
# PET 99.99th percentile: 10693.23662451141
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1801.76998949]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4098.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1285.7676398092242, CT std: 983.5363205324733
# CT 95th percentile: 31.0
# CT 99th percentile: 179.0
# CT 99.9th percentile: 981.0
# CT 99.99th percentile: 1326.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4098.nii.gz
# PET shape: (256, 256, 335), PET_max: 85402.5390625, PET_min: 0.0
# PET mean: 236.1109586323871, PET std: 535.748950442331
# PET 95th percentile: 758.0021759033209
# PET 99th percentile: 1584.8072131347658
# PET 99.9th percentile: 7920.226313964928
# PET 99.99th percentile: 11993.174546775572
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4099.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1364.4413554213916, CT std: 917.5649005889483
# CT 95th percentile: -31.0
# CT 99th percentile: 77.0
# CT 99.9th percentile: 825.0
# CT 99.99th percentile: 1367.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4099.nii.gz
# PET shape: (256, 256, 299), PET_max: 30092.9609375, PET_min: 0.0
# PET mean: 208.07272148375338, PET std: 526.4007199316461
# PET 95th percentile: 937.5985717773438
# PET 99th percentile: 1916.0860461425905
# PET 99.9th percentile: 7636.150505371355
# PET 99.99th percentile: 11042.231298534269
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4102.nii.gz
# CT shape: (512, 512, 527), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1347.30497879231, CT std: 931.3255333445469
# CT 95th percentile: -75.0
# CT 99th percentile: 60.0
# CT 99.9th percentile: 719.0
# CT 99.99th percentile: 3071.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1723.28998995]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4102.nii.gz
# PET shape: (256, 256, 515), PET_max: 56526.2890625, PET_min: 0.0
# PET mean: 115.40361263060295, PET std: 323.91361322867056
# PET 95th percentile: 493.8656005859375
# PET 99th percentile: 1094.3807946777324
# PET 99.9th percentile: 4556.000450195315
# PET 99.99th percentile: 8302.460136230206
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1684.04999018]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4103.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1294.2273176734127, CT std: 977.0235553714954
# CT 95th percentile: 13.0
# CT 99th percentile: 147.0
# CT 99.9th percentile: 952.0
# CT 99.99th percentile: 1361.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4103.nii.gz
# PET shape: (256, 256, 335), PET_max: 17685.927734375, PET_min: 0.0
# PET mean: 205.05291740517958, PET std: 398.94859178636335
# PET 95th percentile: 684.9724975585941
# PET 99th percentile: 1434.715328369141
# PET 99.9th percentile: 5711.576572265658
# PET 99.99th percentile: 8413.50895195191
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4105.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1365.186786154042, CT std: 916.6313672581163
# CT 95th percentile: -44.0
# CT 99th percentile: 73.0
# CT 99.9th percentile: 803.0
# CT 99.99th percentile: 1369.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4105.nii.gz
# PET shape: (256, 256, 299), PET_max: 143189.59375, PET_min: 0.0
# PET mean: 234.91648369551223, PET std: 673.4487781890086
# PET 95th percentile: 952.529724121091
# PET 99th percentile: 2149.5365844726575
# PET 99.9th percentile: 8881.920641601566
# PET 99.99th percentile: 14392.133287986737
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4106.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1346.8224990359915, CT std: 932.9632007628421
# CT 95th percentile: -21.0
# CT 99th percentile: 77.0
# CT 99.9th percentile: 930.0
# CT 99.99th percentile: 1395.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4106.nii.gz
# PET shape: (256, 256, 299), PET_max: 22079.6484375, PET_min: 0.0
# PET mean: 174.10849741651546, PET std: 405.7639117578033
# PET 95th percentile: 732.4144439697259
# PET 99th percentile: 1398.7794274902358
# PET 99.9th percentile: 6052.211055176107
# PET 99.99th percentile: 9114.3673941405
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4114.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1300.2434941805327, CT std: 969.1906248856804
# CT 95th percentile: -15.0
# CT 99th percentile: 152.0
# CT 99.9th percentile: 776.0
# CT 99.99th percentile: 1277.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4114.nii.gz
# PET shape: (256, 256, 299), PET_max: 21689.37890625, PET_min: 0.0
# PET mean: 203.2348637637297, PET std: 491.3831847588703
# PET 95th percentile: 706.4886474609375
# PET 99th percentile: 1570.0377282714858
# PET 99.9th percentile: 7799.845275879117
# PET 99.99th percentile: 11103.634675486766
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4115.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1315.3637030100742, CT std: 958.0790837566346
# CT 95th percentile: -1.0
# CT 99th percentile: 91.0
# CT 99.9th percentile: 619.0
# CT 99.99th percentile: 1275.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4115.nii.gz
# PET shape: (256, 256, 299), PET_max: 16663.388671875, PET_min: 0.0
# PET mean: 265.2747901645243, PET std: 439.2391372564156
# PET 95th percentile: 943.17418823242
# PET 99th percentile: 1723.5818005371098
# PET 99.9th percentile: 5691.7836103520385
# PET 99.99th percentile: 8247.913242675477
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4118.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1336.8988291379999, CT std: 942.4792477264397
# CT 95th percentile: 0.0
# CT 99th percentile: 139.0
# CT 99.9th percentile: 904.0
# CT 99.99th percentile: 1353.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4118.nii.gz
# PET shape: (256, 256, 299), PET_max: 19509.57421875, PET_min: 0.0
# PET mean: 207.25779483955696, PET std: 395.3837653761677
# PET 95th percentile: 870.2860076904287
# PET 99th percentile: 1584.8612060546875
# PET 99.9th percentile: 4765.134108398677
# PET 99.99th percentile: 7922.573280656383
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4120.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1368.7369568355905, CT std: 914.8140142583318
# CT 95th percentile: -21.0
# CT 99th percentile: 88.0
# CT 99.9th percentile: 979.0
# CT 99.99th percentile: 1402.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4120.nii.gz
# PET shape: (256, 256, 299), PET_max: 49883.9375, PET_min: 0.0
# PET mean: 236.803641274872, PET std: 668.445795958679
# PET 95th percentile: 1032.7334777832007
# PET 99th percentile: 2029.4648071289075
# PET 99.9th percentile: 8628.992004883316
# PET 99.99th percentile: 25303.616403120453
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4124.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1332.3383522147562, CT std: 946.9331191641517
# CT 95th percentile: 27.0
# CT 99th percentile: 114.0
# CT 99.9th percentile: 927.0
# CT 99.99th percentile: 1364.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4124.nii.gz
# PET shape: (256, 256, 335), PET_max: 53763.7578125, PET_min: 0.0
# PET mean: 165.8366389907584, PET std: 389.5299614231141
# PET 95th percentile: 668.9000701904304
# PET 99th percentile: 1259.7537841796875
# PET 99.9th percentile: 4866.520665527349
# PET 99.99th percentile: 7651.556695898158
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4125.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1302.7326889151957, CT std: 970.9884826129411
# CT 95th percentile: 39.0
# CT 99th percentile: 136.0
# CT 99.9th percentile: 847.0
# CT 99.99th percentile: 1243.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4125.nii.gz
# PET shape: (256, 256, 335), PET_max: 48412.93359375, PET_min: 0.0
# PET mean: 219.8072830407107, PET std: 487.47719844721456
# PET 95th percentile: 766.4068603515625
# PET 99th percentile: 1558.4411657714859
# PET 99.9th percentile: 6758.029874023934
# PET 99.99th percentile: 10547.27628388524
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4128.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1264.55637348232, CT std: 998.7789364863995
# CT 95th percentile: 24.0
# CT 99th percentile: 155.0
# CT 99.9th percentile: 889.0
# CT 99.99th percentile: 3071.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4128.nii.gz
# PET shape: (256, 256, 335), PET_max: 13001.5830078125, PET_min: 0.0
# PET mean: 216.7649609166699, PET std: 346.3480001750249
# PET 95th percentile: 692.3860473632812
# PET 99th percentile: 1339.585948486329
# PET 99.9th percentile: 4548.665208984421
# PET 99.99th percentile: 6824.138555711803
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4129.nii.gz
# CT shape: (512, 512, 551), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1381.3928333566323, CT std: 900.1387195032715
# CT 95th percentile: -94.0
# CT 99th percentile: 56.0
# CT 99.9th percentile: 578.0
# CT 99.99th percentile: 1256.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1801.76998949]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4129.nii.gz
# PET shape: (256, 256, 551), PET_max: 21370.171875, PET_min: 0.0
# PET mean: 129.18758347649015, PET std: 413.90587463195016
# PET 95th percentile: 587.8262329101562
# PET 99th percentile: 1447.6973022460934
# PET 99.9th percentile: 6282.273398438294
# PET 99.99th percentile: 10858.397425292627
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1801.76998949]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4130.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1338.983704250962, CT std: 940.8265012574082
# CT 95th percentile: 19.0
# CT 99th percentile: 98.0
# CT 99.9th percentile: 883.0
# CT 99.99th percentile: 1364.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4130.nii.gz
# PET shape: (256, 256, 335), PET_max: 16724.40234375, PET_min: 0.0
# PET mean: 145.84772674459148, PET std: 346.9992604024683
# PET 95th percentile: 610.8650390625007
# PET 99th percentile: 1214.9207067871098
# PET 99.9th percentile: 4710.940124512335
# PET 99.99th percentile: 10317.275576269396
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4131.nii.gz
# CT shape: (512, 512, 299), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1330.4228212362946, CT std: 946.484686777922
# CT 95th percentile: -9.0
# CT 99th percentile: 97.0
# CT 99.9th percentile: 877.0
# CT 99.99th percentile: 1372.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [700.00024414 700.00024414 977.7299943 ]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4131.nii.gz
# PET shape: (256, 256, 299), PET_max: 20839.615234375, PET_min: 0.0
# PET mean: 149.1429083087325, PET std: 426.5597805228373
# PET 95th percentile: 570.8599060058591
# PET 99th percentile: 1219.1787902832089
# PET 99.9th percentile: 7043.540177734598
# PET 99.99th percentile: 10744.42758378863
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [600.        600.        977.7299943]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4134.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1352.585165223079, CT std: 928.7272787572989
# CT 95th percentile: 7.0
# CT 99th percentile: 98.0
# CT 99.9th percentile: 885.0
# CT 99.99th percentile: 1428.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4134.nii.gz
# PET shape: (256, 256, 335), PET_max: 17405.794921875, PET_min: 0.0
# PET mean: 183.55411834578987, PET std: 352.8705311378955
# PET 95th percentile: 817.5810546875
# PET 99th percentile: 1518.7327563476565
# PET 99.9th percentile: 3907.526432617309
# PET 99.99th percentile: 5704.335362499929
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4137.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1310.0690610971023, CT std: 962.8554429924458
# CT 95th percentile: 7.0
# CT 99th percentile: 94.0
# CT 99.9th percentile: 851.0
# CT 99.99th percentile: 1268.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4137.nii.gz
# PET shape: (256, 256, 335), PET_max: 17716.029296875, PET_min: 0.0
# PET mean: 224.48653645599796, PET std: 411.3140398203459
# PET 95th percentile: 862.6934234619141
# PET 99th percentile: 1629.4930627441413
# PET 99.9th percentile: 5518.444669922079
# PET 99.99th percentile: 8162.338684373652
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4138.nii.gz
# CT shape: (512, 512, 335), CT_max: 2448.0, CT_min: -3024.0
# CT mean: -1349.6262273759985, CT std: 931.5822291953656
# CT 95th percentile: 17.0
# CT 99th percentile: 89.0
# CT 99.9th percentile: 885.0
# CT 99.99th percentile: 1336.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4138.nii.gz
# PET shape: (256, 256, 335), PET_max: 945832.5625, PET_min: 0.0
# PET mean: 194.9922368764716, PET std: 992.9462171354199
# PET 95th percentile: 796.4364959716802
# PET 99th percentile: 1660.2335766601564
# PET 99.9th percentile: 7286.264921875263
# PET 99.99th percentile: 10813.43200956974
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# File: CTACIVV_4139.nii.gz
# CT shape: (512, 512, 335), CT_max: 3071.0, CT_min: -3024.0
# CT mean: -1331.641331800774, CT std: 946.0298013738951
# CT 95th percentile: 7.0
# CT 99th percentile: 151.0
# CT 99.9th percentile: 882.0
# CT 99.99th percentile: 1432.0
# CT physcial spacing: (1.367188, 1.367188, 3.27)
# CT physical range: [ 700.00024414  700.00024414 1095.44999361]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# File: PET_TOFNAC_E4139.nii.gz
# PET shape: (256, 256, 335), PET_max: 19664.7734375, PET_min: 0.0
# PET mean: 176.7433968582235, PET std: 340.4379175663879
# PET 95th percentile: 737.1355224609379
# PET 99th percentile: 1362.3043920898444
# PET 99.9th percentile: 3971.2982170410296
# PET 99.99th percentile: 7050.004531536575
# PET physcial spacing: (2.34375, 2.34375, 3.27)
# PET physical range: [ 600.          600.         1095.44999361]
# <--->
# root@fac5e29efbe7:/SharkSeagrass# 
# root@fac5e29efbe7:/SharkSeagrass# 

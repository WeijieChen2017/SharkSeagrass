tag_list = [
    'E4055', 'E4058', 'E4061', 'E4066', 'E4068',
    'E4069', 'E4073', 'E4074', 'E4077', 'E4078',
    'E4079', 'E4081', 'E4084', 'E4091', 'E4092',
    'E4094', 'E4096', 'E4098', 'E4099', 'E4103',
    'E4105', 'E4106', 'E4114', 'E4115', 'E4118',
    'E4120', 'E4124', 'E4125', 'E4128', 'E4129',
    'E4130', 'E4131', 'E4134', 'E4137', 'E4138',
    'E4139', 'E4143', 'E4144', 'E4147', 'E4152',
    'E4155', 'E4157', 'E4158', 'E4162', 'E4163',
    'E4165', 'E4166', 'E4172', 'E4181', 'E4182',
    'E4183', 'E4185', 'E4187', 'E4189', 'E4193',
    'E4197', 'E4198', 'E4207', 'E4208', 'E4216',
    'E4217', 'E4219', 'E4220', 'E4232', 'E4237',
    'E4238', 'E4239', 'E4241',
]

root_folder = "James_data_v3/"

# randomly divide the tag_list into 5 parts from cv0 to cv4
import random

random.seed(0)

random.shuffle(tag_list)
total_len = len(tag_list)
cv_len = total_len // 5
cv_list = []
for i in range(5):
    if i == 4:
        cv_list.append(tag_list[i * cv_len:])
    else:
        cv_list.append(tag_list[i * cv_len: (i + 1) * cv_len])

print("cv_list: ", cv_list)

json_dict = {
    "cv0": cv_list[0],
    "cv1": cv_list[1],
    "cv2": cv_list[2],
    "cv3": cv_list[3],
    "cv4": cv_list[4],
}

import json

with open(root_folder + "cv_list.json", "w") as f:
    json.dump(json_dict, f)

print("cv_list.json saved.")

import os

# cmd_1 = "python UNetUNet_v1_py5_step6_eval.py -c 0"
cmd_2 = "python UNetUNet_v1_py5_step6_eval.py -c 1"
cmd_3 = "python UNetUNet_v1_py5_step6_eval.py -c 2"
cmd_4 = "python UNetUNet_v1_py5_step6_eval.py -c 3"
cmd_5 = "python UNetUNet_v1_py5_step6_eval.py -c 4"

for cmd in [cmd_2, cmd_3, cmd_4, cmd_5]:
    print("<<"*20)
    print(cmd)
    os.system(cmd)
    print(">>"*20)
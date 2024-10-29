# input is a list of 3D images and 3D masks (001.npy and 001_mask.npy)
# shuffle them into training/validation/testing sets according to the given ratio
# save the shuffled list into a json file

import os
import numpy as np
import json

def shuffle_data(data_list, ratio, save_folder):
    
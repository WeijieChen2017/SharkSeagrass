import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix
from skimage.metrics import mean_squared_error

def denorm_CT(data):
    data *= 4000
    data -= 1024
    return data

def rmse(x,y):
    return np.mean(np.sqrt(np.sum(np.square(x-y))))

def nrmse(x,y):
    # compute the normalized root mean squared error
    return rmse(x,y) / (np.max(x) - np.min(x))

def mse(x,y):
    return mean_squared_error(x,y)

def mae(x,y):
    return np.mean(np.absolute(x-y))

def acutance(x):
    return np.mean(np.absolute(sobel(x)))

def dice_coe(x, y, tissue="air"):
    if tissue == "air":
        x_mask = filter_data(x, -2000, -500)
        y_mask = filter_data(y, -2000, -500)
    if tissue == "soft":
        x_mask = filter_data(x, -500, 250)
        y_mask = filter_data(y, -500, 250)
    if tissue == "bone":
        x_mask = filter_data(x, 250, 3000)
        y_mask = filter_data(y, 250, 3000)
    CM = confusion_matrix(np.ravel(x_mask), np.ravel(y_mask))
    TN, FP, FN, TP = CM.ravel()
    return 2*TP / (2*TP + FN + FP)

def filter_data(data, range_min, range_max):
    mask_1 = data < range_max
    mask_2 = data > range_min
    mask_1 = mask_1.astype(int)
    mask_2 = mask_2.astype(int)
    mask = mask_1 * mask_2
    return mask


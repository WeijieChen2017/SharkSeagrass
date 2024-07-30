import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

PET_file = "synCT_PET_James/ori/crop/E4055_PET_crop_th04.nii.gz"

PET_file = nib.load(PET_file)
PET_data = PET_file.get_fdata()
print("Loaded PET data shape: ", PET_data.shape)

# randomly cut 64*64*64 cubes from the PET data
cut_size = 64
num_cubes = 10
num_images = 5

width = num_images * 2
height = 3

for idx_img in range(num_images):


    plt.figure(figsize=(width, height), dpi=200)

    for idx_cub in range(num_cubes):
        
        # randomly select the center of the cube
        x = np.random.randint(0, PET_data.shape[0] - cut_size)
        y = np.random.randint(0, PET_data.shape[1] - cut_size)
        z = np.random.randint(0, PET_data.shape[2] - cut_size)

        cut_cube = PET_data[x:x+cut_size, y:y+cut_size, z:z+cut_size]
        PET_mean = np.mean(cut_cube)

        # set the subplot for 3 rows and num_cubes columns
        plt.subplot(3, num_cubes, idx_cub+1)
        plt.imshow(cut_cube[:, :, cut_size//2], cmap="gray")
        plt.axis("off")
        plt.title(f"{PET_mean:.4f}")

        plt.subplot(3, num_cubes, num_cubes+idx_cub+1)
        plt.imshow(cut_cube[:, cut_size//2, :], cmap="gray")
        plt.axis("off")
        plt.title(f"{PET_mean:.4f}")

        plt.subplot(3, num_cubes, 2*num_cubes+idx_cub+1)
        plt.imshow(cut_cube[cut_size//2, :, :], cmap="gray")
        plt.axis("off")
        plt.title(f"{PET_mean:.4f}")

    plt.tight_layout()
    save_name = f"synCT_PET_James/ori/crop/cut_cubes_{idx_img}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"[{idx_img+1}/{num_images}] Saved at {save_name}")



# after the observation, we say PET_mean < 0.01 is discarded
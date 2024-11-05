m0_path = "~/Downloads/E4058/m0.nii.gz"
m1_path = "~/Downloads/E4058/m1.nii.gz"
m2_path = "~/Downloads/E4058/m2.nii.gz"
m3_path = "~/Downloads/E4058/m3.nii.gz"
m4_path = "~/Downloads/E4058/m4.nii.gz"

import os
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt

m0 = nib.load(os.path.expanduser(m0_path)).get_fdata()
m1 = nib.load(os.path.expanduser(m1_path)).get_fdata()
m2 = nib.load(os.path.expanduser(m2_path)).get_fdata()
m3 = nib.load(os.path.expanduser(m3_path)).get_fdata()
m4 = nib.load(os.path.expanduser(m4_path)).get_fdata()

print(m0.shape, m1.shape, m2.shape, m3.shape, m4.shape)

lenz = m0.shape[2]
m1 = m1[:, :, 0:lenz]
m2 = m2[:, :, 0:lenz]
m3 = m3[:, :, 0:lenz]
m4 = m4[:, :, 0:lenz]

m_data = {
    "m0": m0,
    "m1": m1,
    "m2": m2,
    "m3": m3,
    "m4": m4,
}

# axial
idx_z = 278
save_path = f"~/Downloads/E4058/idz{idx_z}.png"
plt.figure(figsize=(10, 10), dpi=300)

for i in range(4):
    plt.subplot(1, 5, i + 1)
    img = m_data[f"m{i}"][:, :, idx_z]
    img = np.rot90(img, 1)
    plt.imshow(img, cmap="gray", vmin=-1024, vmax=600)
    plt.axis("off")

plt.subplot(1, 5, 5)
img = m_data["m4"][:, :, idx_z]
img = np.rot90(img, 1)
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.)
plt.savefig(os.path.expanduser(save_path))

# sagittal
idx_x = 132
save_path = f"~/Downloads/E4058/idx{idx_x}.png"
plt.figure(figsize=(10, 10), dpi=300)

for i in range(4):
    plt.subplot(1, 5, i + 1)
    img = m_data[f"m{i}"][idx_x, :, :]
    img = np.rot90(img, 1)
    # flip horizontally
    img = np.fliplr(img)
    plt.imshow(img, cmap="gray", vmin=-1024, vmax=600)
    plt.axis("off")

plt.subplot(1, 5, 5)
img = m_data["m4"][idx_x, :, :]
img = np.rot90(img, 1)
img = np.fliplr(img)
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.)
plt.savefig(os.path.expanduser(save_path))

# coronal
idx_y = 93
save_path = f"~/Downloads/E4058/idy{idx_y}.png"
plt.figure(figsize=(10, 10), dpi=300)

for i in range(4):
    plt.subplot(1, 5, i + 1)
    img = m_data[f"m{i}"][:, idx_y, :]
    img = np.rot90(img, 1)
    # flip horizontally
    img = np.fliplr(img)
    plt.imshow(img, cmap="gray", vmin=-1024, vmax=600)
    plt.axis("off")

plt.subplot(1, 5, 5)
img = m_data["m4"][:, idx_y, :]
img = np.rot90(img, 1)
img = np.fliplr(img)
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.subplots_adjust(wspace=0.)
plt.savefig(os.path.expanduser(save_path))
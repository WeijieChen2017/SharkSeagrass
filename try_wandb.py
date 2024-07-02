import os

# Define the base cache directory
base_cache_dir = './cache'

# Define and create necessary subdirectories within the base cache directory
cache_dirs = {
    'WANDB_DIR': os.path.join(base_cache_dir, 'wandb'),
    'WANDB_CACHE_DIR': os.path.join(base_cache_dir, 'wandb_cache'),
    'WANDB_CONFIG_DIR': os.path.join(base_cache_dir, 'config'),
    'TRANSFORMERS_CACHE': os.path.join(base_cache_dir, 'transformers'),
    'MPLCONFIGDIR': os.path.join(base_cache_dir, 'mplconfig')
}

# Create the base cache directory if it doesn't exist
os.makedirs(base_cache_dir, exist_ok=True)

# Create the necessary subdirectories and set the environment variables
for key, path in cache_dirs.items():
    os.makedirs(path, exist_ok=True)
    os.environ[key] = path

# Now you can use these directories with WandB
import wandb

# Initialize WandB with the updated environment variables

wandb.login(key = "41c33ee621453a8afcc7b208674132e0e8bfafdb")
wandb_run = wandb.init(project="try_wandb", dir=os.getenv("WANDB_DIR", "cache/wandb"),
    config={
        "msg": "Hello from WandB!",
        "WANDB_DIR": os.getenv("WANDB_DIR"),
        "WANDB_CACHE_DIR": os.getenv("WANDB_CACHE_DIR"),
        "WANDB_CONFIG_DIR": os.getenv("WANDB_CONFIG_DIR"),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
        "MPLCONFIGDIR": os.getenv("MPLCONFIGDIR")
        }
)

import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=None, wandb_commitment=True):
    # numpy_x = x[0, :, :, :, :].cpu().numpy().squeeze()
    # numpy_xrec = xrec[0, :, :, :, :].cpu().numpy().squeeze()
    x = x[0, 0, :, :, :]
    xrec = xrec[0, 0, :, :, :]
    x_clip = np.clip(x, 0, 1)
    rec_clip = np.clip(xrec, 0, 1)
    fig_width = num_per_direction * 3
    fig_height = 4
    fig, axs = plt.subplots(3, fig_width, figsize=(fig_width, fig_height), dpi=100)
    # for axial
    for i in range(num_per_direction):
        img_x = x_clip[x_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        img_rec = rec_clip[rec_clip.shape[0]//(num_per_direction+1)*(i+1), :, :]
        axs[0, 3*i].imshow(img_x, cmap="gray")
        axs[0, 3*i].set_title(f"A x {x_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i].axis("off")
        axs[1, 3*i].imshow(img_rec, cmap="gray")
        axs[1, 3*i].set_title(f"A xrec {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i].axis("off")
        axs[2, 3*i].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i].set_title(f"A diff {rec_clip.shape[0]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i].axis("off")
    # for sagittal
    for i in range(num_per_direction):
        img_x = x_clip[:, :, x_clip.shape[2]//(num_per_direction+1)*(i+1)]
        img_rec = rec_clip[:, :, rec_clip.shape[2]//(num_per_direction+1)*(i+1)]
        axs[0, 3*i+1].imshow(img_x, cmap="gray")
        axs[0, 3*i+1].set_title(f"S x {x_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+1].axis("off")
        axs[1, 3*i+1].imshow(img_rec, cmap="gray")
        axs[1, 3*i+1].set_title(f"S xrec {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+1].axis("off")
        axs[2, 3*i+1].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+1].set_title(f"S diff {rec_clip.shape[2]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+1].axis("off")

    # for coronal
    for i in range(num_per_direction):
        img_x = x_clip[:, x_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        img_rec = rec_clip[:, rec_clip.shape[1]//(num_per_direction+1)*(i+1), :]
        axs[0, 3*i+2].imshow(img_x, cmap="gray")
        axs[0, 3*i+2].set_title(f"C x {x_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[0, 3*i+2].axis("off")
        axs[1, 3*i+2].imshow(img_rec, cmap="gray")
        axs[1, 3*i+2].set_title(f"C xrec {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[1, 3*i+2].axis("off")
        axs[2, 3*i+2].imshow(img_x - img_rec, cmap="bwr")
        axs[2, 3*i+2].set_title(f"C diff {rec_clip.shape[1]//(num_per_direction+1)*(i+1)}")
        axs[2, 3*i+2].axis("off")

    plt.tight_layout()
    plt.savefig(savename)
    wandb_run.log({"val_snapshots": fig}, commit=wandb_commitment)
    plt.close()
    print(f"Save the plot to {savename}")


# x, xrec: (B, C, D, H, W) = 1, 1, 64, 64, 64
# create random x and xrec for testing

x = np.random.rand(1, 1, 64, 64, 64)
xrec = np.random.rand(1, 1, 64, 64, 64)

# try 1 image per direction
# save_name = "x_xrec_1.png"
# plot_and_save_x_xrec(x, xrec, num_per_direction=1, savename=save_name, wandb_commitment=False)
# wandb.save(f"{save_name}.png", base_path="./", policy="now")
# wandb.upload_file(f"{save_name}.png", root=".")
# wandb.log({"val_snapshots": wandb.Image(f"{save_name}")})

# try 2 images per direction
# save_name = "x_xrec_2.png"
# plot_and_save_x_xrec(x, xrec, num_per_direction=2, savename=save_name, wandb_commitment=False)
# wandb.upload_file(f"{save_name}.png", root=".")
# wandb.log({"val_snapshots": wandb.Image(f"{save_name}")})

# try 3 images per direction
save_name = "x_xrec_3.png"
plot_and_save_x_xrec(x, xrec, num_per_direction=3, savename=save_name)
# wandb.upload_file(f"{save_name}.png", root=".")
# wandb.log({"val_snapshots": wandb.Image(f"{save_name}")})

# try histogram
wandb_run.log({"gradients": wandb.Histogram(np.ravel(x))})

print("Done!")
wandb.finish()
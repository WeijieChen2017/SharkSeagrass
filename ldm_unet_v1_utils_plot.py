import numpy as np
import matplotlib.pyplot as plt
import os

CT_NORM = 5000
CT_MIN = -1024
CT_MAX = 3976

mode = "f(x)+x->y" # model output + input -> ground truth
# outputs is 
mode = "f(x)->y-x" # model output -> ground truth - input

def inputs_labels_outputs_to_imgs(inputs, labels, outputs, cube_size, cut_index, i):

    if cut_index == "z":
        img_PET = np.rot90(inputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1)) # -1 to 1
       
        img_CT = np.rot90(labels[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1)) # -1 to 1
        
        img_pred = np.rot90(outputs[i, 0, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1)) # -1 to 1

    elif cut_index == "y":
        img_PET = np.rot90(inputs[i, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))

        img_CT = np.rot90(labels[i, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))

        img_pred = np.rot90(outputs[i, 0, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))

    elif cut_index == "x":

        img_PET = np.rot90(inputs[i, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))

        img_CT = np.rot90(labels[i, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))

        img_pred = np.rot90(outputs[i, 0, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
    
    else:
        raise ValueError("cut_index must be either x, y, or z")

    img_5 = img_pred + img_PET # -1 to 1
    img_5 = (img_5 + 1) / 2 # 0 to 1
    img_1 = (img_PET + 1) / 2 # 0 to 1
    img_2 = (img_CT + 1) / 2 # 0 to 1
    img_3 = (img_pred + 1) / 2 # 0 to 1
    img_4 = img_2 - img_1 # -1 to 1
    img_4 = (img_4 + 1) / 2 # 0 to 1
    
    return img_1, img_2, img_3, img_4, img_5

def plot_results(inputs, labels, outputs, idx_epoch, root_folder, cube_size):
    # plot the results
    n_block = 8
    if inputs.shape[0] < n_block:
        n_block = inputs.shape[0]
    fig = plt.figure(figsize=(12, n_block*3.6), dpi=300)

    n_row = n_block * 3
    n_col = 10

    # compute mean for img_PET
    img_PET_mean = np.mean(inputs.detach().cpu().numpy(), axis=(1, 2, 3, 4))
    img_PET_mean = (img_PET_mean + 1) / 2
    img_PET_mean = img_PET_mean * CT_NORM + CT_MIN
    fig.suptitle(f"Epoch {idx_epoch}, mean PET: {img_PET_mean}", fontsize=16)

    # for axial view

    for i_asc in range(3): # for axial, sagittal, coronal

        if i_asc == 0:
            cut_index = "z"
        elif i_asc == 1:
            cut_index = "y"
        elif i_asc == 2:
            cut_index = "x"
        
        for i in range(n_block):

            img_1, img_2, img_3, img_4, img_5 = inputs_labels_outputs_to_imgs(inputs, labels, outputs, cube_size, cut_index, i)
            
            # first three and hist
            plt.subplot(n_row, n_col, i * n_col + 1 + i_asc * 10)
            plt.imshow(img_1, cmap="gray", vmin=0, vmax=0.5) # x
            # plt.title("input PET")
            if i == 0:
                plt.title("input STEP1")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 2 + i_asc * 10)
            plt.imshow(img_2, cmap="gray", vmin=0, vmax=0.5) # y
            # plt.title("label CT")
            if i == 0:
                plt.title("input STEP2")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 3 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_3, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
            # plt.title("output CT")
            plt.colorbar()
            if i == 0:
                plt.title("f(x)=yhat-x")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 4 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_4, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
            plt.colorbar()
            if i == 0:
                plt.title("gt=y-x")
            # plt.title("output CT")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 5 + i_asc * 10)
            # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
            plt.imshow(img_5, cmap="gray", vmin=0, vmax=0.5) # yhat
            if i == 0:
                plt.title("yhat")
            # plt.title("output CT")
            plt.axis("off")

            plt.subplot(n_row, n_col, i * n_col + 6 + i_asc * 10)
            # img_PET = np.clip(img_PET, 0, 1)
            plt.hist(img_1.flatten(), bins=100)
            # plt.title("input PET")
            if i == 0:
                plt.title("input STEP1")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)

            plt.subplot(n_row, n_col, i * n_col + 7 + i_asc * 10)
            # img_CT = np.clip(img_CT, 0, 1)
            plt.hist(img_2.flatten(), bins=100)
            # plt.title("label CT")
            if i == 0:
                plt.title("input STEP2")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)

            plt.subplot(n_row, n_col, i * n_col + 8 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_5 - img_1).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("f(x)=yhat-x")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(-1, 1)

            plt.subplot(n_row, n_col, i * n_col + 9 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_2 - img_1).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("gt=y-x")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(-1, 1)

            plt.subplot(n_row, n_col, i * n_col + 10 + i_asc * 10)
            # img_pred = np.clip(img_pred, 0, 1)
            plt.hist((img_5).flatten(), bins=100)
            # plt.title("output CT")
            if i == 0:
                plt.title("yhat")
            plt.yscale("log")
            # plt.axis("off")
            plt.xlim(0, 1)


    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"epoch_{idx_epoch}.png"))
    plt.close()
import numpy as np
import matplotlib.pyplot as plt
import os





def plot_results(inputs, labels, outputs, idx_epoch, root_folder, cube_size):
    # plot the results
    n_block = 8
    if inputs.shape[0] < n_block:
        n_block = inputs.shape[0]
    plt.figure(figsize=(12, n_block*3.6), dpi=300)

    n_row = n_block * 3
    n_col = 10

    # for axial view

    for i in range(n_block):

        img_PET = np.rot90(inputs[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))
        img_PET = (img_PET + 1) / 2

        img_CT = np.rot90(labels[i, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))
        img_CT = (img_CT + 1) / 2

        img_pred = np.rot90(outputs[i, 0, :, :, :, cube_size // 2].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
        img_pred = (img_pred + 1) / 2

        yhat_x = img_pred - img_PET # -1 to 1
        yhat_x = (yhat_x + 1) / 2 # 0 to 1

        y_x = img_CT - img_PET
        y_x = (y_x + 1) / 2

        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.imshow(img_PET, cmap="gray", vmin=0, vmax=0.5) # x
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.imshow(img_CT, cmap="gray", vmin=0, vmax=0.5) # y
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(yhat_x, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
        # plt.title("output CT")
        plt.colorbar()
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(y_x, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
        plt.colorbar()
        if i == 0:
            plt.title("gt=y-x")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 5)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(img_pred, cmap="gray", vmin=0, vmax=0.5) # yhat
        if i == 0:
            plt.title("yhat")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 6)
        # img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 7)
        # img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 8)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 9)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_CT-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("gt=y-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 10)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("yhat")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)


    # for sagittal view

    for ii in range(n_block):
        
        i = ii + n_block * n_col
        img_PET = np.rot90(inputs[ii, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))
        img_PET = (img_PET + 1) / 2

        img_CT = np.rot90(labels[ii, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))
        img_CT = (img_CT + 1) / 2

        img_pred = np.rot90(outputs[ii, 0, :, :, cube_size // 2, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
        img_pred = (img_pred + 1) / 2

        yhat_x = img_pred - img_PET # -1 to 1
        yhat_x = (yhat_x + 1) / 2 # 0 to 1

        y_x = img_CT - img_PET
        y_x = (y_x + 1) / 2

        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.imshow(img_PET, cmap="gray", vmin=0, vmax=0.5) # x
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.imshow(img_CT, cmap="gray", vmin=0, vmax=0.5) # y
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(yhat_x, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
        # plt.title("output CT")
        plt.colorbar()
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(y_x, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
        plt.colorbar()
        if i == 0:
            plt.title("gt=y-x")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 5)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(img_pred, cmap="gray", vmin=0, vmax=0.5) # yhat
        if i == 0:
            plt.title("yhat")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 6)
        # img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 7)
        # img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 8)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 9)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_CT-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("gt=y-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 10)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("yhat")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)


    # for coronal view

    for iii in range(n_block):
        
        i = iii + n_block * n_col * 2
        img_PET = np.rot90(inputs[iii, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_PET = np.squeeze(np.clip(img_PET, -1, 1))
        img_PET = (img_PET + 1) / 2

        img_CT = np.rot90(labels[iii, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_CT = np.squeeze(np.clip(img_CT, -1, 1))
        img_CT = (img_CT + 1) / 2

        img_pred = np.rot90(outputs[iii, 0, :, cube_size // 2, :, :].detach().cpu().numpy())
        img_pred = np.squeeze(np.clip(img_pred, -1, 1))
        img_pred = (img_pred + 1) / 2

        yhat_x = img_pred - img_PET # -1 to 1
        yhat_x = (yhat_x + 1) / 2 # 0 to 1

        y_x = img_CT - img_PET
        y_x = (y_x + 1) / 2

        # first three and hist
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.imshow(img_PET, cmap="gray", vmin=0, vmax=0.5) # x
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.imshow(img_CT, cmap="gray", vmin=0, vmax=0.5) # y
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 3)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(yhat_x, cmap="bwr", vmin=0.45, vmax=0.55) # yhat = f(x) + x, img_pred = f(x) = yhat - x
        # plt.title("output CT")
        plt.colorbar()
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 4)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(y_x, cmap="bwr", vmin=0.45, vmax=0.55) # y = x + (y - x), (y - x) = y - x
        plt.colorbar()
        if i == 0:
            plt.title("gt=y-x")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 5)
        # outputs.shape:  torch.Size([16, 2, 1, 400, 400])
        plt.imshow(img_pred, cmap="gray", vmin=0, vmax=0.5) # yhat
        if i == 0:
            plt.title("yhat")
        # plt.title("output CT")
        plt.axis("off")

        plt.subplot(n_row, n_col, i * n_col + 6)
        # img_PET = np.clip(img_PET, 0, 1)
        plt.hist(img_PET.flatten(), bins=100)
        # plt.title("input PET")
        if i == 0:
            plt.title("input STEP1")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 7)
        # img_CT = np.clip(img_CT, 0, 1)
        plt.hist(img_CT.flatten(), bins=100)
        # plt.title("label CT")
        if i == 0:
            plt.title("input STEP2")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

        plt.subplot(n_row, n_col, i * n_col + 8)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("f(x)=yhat-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 9)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_CT-img_PET).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("gt=y-x")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(-1, 1)

        plt.subplot(n_row, n_col, i * n_col + 10)
        # img_pred = np.clip(img_pred, 0, 1)
        plt.hist((img_pred).flatten(), bins=100)
        # plt.title("output CT")
        if i == 0:
            plt.title("yhat")
        plt.yscale("log")
        # plt.axis("off")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"epoch_{idx_epoch}.png"))
    plt.close()
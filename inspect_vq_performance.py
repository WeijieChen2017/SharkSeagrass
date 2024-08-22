import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

vq_list = ["f4", "f4-noattn", "f8", "f8-n256", "f16"]

tag_list = [
    "E4055", "E4058", "E4061",          "E4066",
    "E4068", "E4069", "E4073", "E4074", "E4077",
    "E4078", "E4079",          "E4081", "E4084",
             "E4091", "E4092", "E4094", "E4096",
             "E4098", "E4099",          "E4103",
    "E4105", "E4106", "E4114", "E4115", "E4118",
    "E4120", "E4124", "E4125", "E4128", "E4129",
    "E4130", "E4131", "E4134", "E4137", "E4138",
    "E4139",
]

# for vq_tag in vq_list:
#     print()
#     print("Processing VQ:", vq_tag)
#     PET_loss_list = []
#     CTr_loss_list = []
#     PET_indices_list = []
#     CTr_indices_list = []
#     for case_tag in tag_list:
#         PET_loss_path = f"./B100/vq_{vq_tag}_loss/vq_{vq_tag}_{case_tag}_PET_l1_loss.npy"
#         CTr_loss_path = f"./B100/vq_{vq_tag}_loss/vq_{vq_tag}_{case_tag}_CTr_l1_loss.npy"
#         PET_loss = np.load(PET_loss_path)
#         CTr_loss = np.load(CTr_loss_path)
#         # print(f"Loaded PET loss: {PET_loss_path}, with the shape {PET_loss.shape}")
#         # print(f"Loaded CTr loss: {CTr_loss_path}, with the shape {CTr_loss.shape}")
#         # exit()
#         PET_loss_list.append(PET_loss.mean())
#         CTr_loss_list.append(CTr_loss.mean())

#         PET_ind_path = f"./B100/vq_{vq_tag}_ind/vq_{vq_tag}_{case_tag}_PET_ind.npy"
#         CTr_ind_path = f"./B100/vq_{vq_tag}_ind/vq_{vq_tag}_{case_tag}_CTr_ind.npy"
#         PET_indices = np.load(PET_ind_path)
#         CTr_indices = np.load(CTr_ind_path)
#         # count how many indices are used
#         PET_indices_list.append(np.unique(PET_indices).shape[0])
#         CTr_indices_list.append(np.unique(CTr_indices).shape[0])

#     PET_loss_list = np.array(PET_loss_list)
#     CTr_loss_list = np.array(CTr_loss_list)
#     PET_loss_mean = PET_loss_list.mean()
#     CTr_loss_mean = CTr_loss_list.mean()
#     print(f"VQ: {vq_tag}, PET loss mean: {PET_loss_mean:.3f}, CTr loss mean: {CTr_loss_mean:.3f}")

#     PET_indices_list = np.array(PET_indices_list)
#     CTr_indices_list = np.array(CTr_indices_list)
#     PET_indices_mean = PET_indices_list.mean()
#     CTr_indices_mean = CTr_indices_list.mean()
#     vq_config_path = f"ldm_models/first_stage_models/vq-{vq_tag}/config.yaml"
#     with open(vq_config_path, "r") as f:
#         vq_config = yaml.load(f, Loader=yaml.FullLoader)
#     vq_embed_dim = vq_config["model"]["params"]["embed_dim"]
#     vq_n_embed = vq_config["model"]["params"]["n_embed"]

#     print(f"VQ: {vq_tag}, PET indices mean: {PET_indices_mean:.3f} out of {vq_n_embed} ({PET_indices_mean / vq_n_embed * 100:.2f}%) at {vq_embed_dim} dimensions")
#     print(f"VQ: {vq_tag}, CTr indices mean: {CTr_indices_mean:.3f} out of {vq_n_embed} ({CTr_indices_mean / vq_n_embed * 100:.2f}%) at {vq_embed_dim} dimensions")

#     print()


# Processing VQ: f4
# VQ: f4, PET loss mean: 119.559, CTr loss mean: 32.521
# VQ: f4, PET indices mean: 3868.278 out of 8192 (47.22%) at 3 dimensions
# VQ: f4, CTr indices mean: 3843.944 out of 8192 (46.92%) at 3 dimensions

# Processing VQ: f4-noattn
# VQ: f4-noattn, PET loss mean: 78.551, CTr loss mean: 12.890
# VQ: f4-noattn, PET indices mean: 2819.139 out of 8192 (34.41%) at 3 dimensions
# VQ: f4-noattn, CTr indices mean: 2807.833 out of 8192 (34.28%) at 3 dimensions

# Processing VQ: f8
# VQ: f8, PET loss mean: 123.766, CTr loss mean: 25.625
# VQ: f8, PET indices mean: 879.944 out of 16384 (5.37%) at 4 dimensions
# VQ: f8, CTr indices mean: 868.611 out of 16384 (5.30%) at 4 dimensions

# Processing VQ: f8-n256
# VQ: f8-n256, PET loss mean: 142.058, CTr loss mean: 31.403
# VQ: f8-n256, PET indices mean: 256.000 out of 256 (100.00%) at 4 dimensions
# VQ: f8-n256, CTr indices mean: 255.889 out of 256 (99.96%) at 4 dimensions

# Processing VQ: f16
# VQ: f16, PET loss mean: 171.096, CTr loss mean: 26.961
# VQ: f16, PET indices mean: 11958.278 out of 16384 (72.99%) at 8 dimensions
# VQ: f16, CTr indices mean: 12166.556 out of 16384 (74.26%) at 8 dimensions


save_folder = "./B100/"

for vq_tag in vq_list:
    print()
    print("Processing VQ:", vq_tag)
    vq_config_path = f"ldm_models/first_stage_models/vq-{vq_tag}/config.yaml"
    with open(vq_config_path, "r") as f:
        vq_config = yaml.load(f, Loader=yaml.FullLoader)
    vq_embed_dim = vq_config["model"]["params"]["embed_dim"]
    vq_n_embed = vq_config["model"]["params"]["n_embed"]
    PET_CT_indices_correration = np.zeros((vq_n_embed, vq_n_embed))
    for case_tag in tag_list:
        PET_ind_path = f"./B100/vq_{vq_tag}_ind/vq_{vq_tag}_{case_tag}_PET_ind.npy"
        CTr_ind_path = f"./B100/vq_{vq_tag}_ind/vq_{vq_tag}_{case_tag}_CTr_ind.npy"
        PET_indices = np.load(PET_ind_path)
        CTr_indices = np.load(CTr_ind_path)
        for PET_ind in PET_indices:
            for CTr_ind in CTr_indices:
                PET_CT_indices_correration[PET_ind, CTr_ind] += 1

    # for each PET index, compute the CTr index probability
    PET_CT_indices_correration = PET_CT_indices_correration / PET_CT_indices_correration.sum(axis=1, keepdims=True)

    # plot the heat map
    plt.figure()
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(PET_CT_indices_correration, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(f"VQ: {vq_tag}, PET and CTr indices correlation")
    plt.xlabel("CTr indices")
    plt.ylabel("PET indices")
    plt.axis("equal")
    plt.savefig(f"{save_folder}/vq_{vq_tag}_indices_correration.png")
    plt.close()
    print(f"Saved correlation plot {save_folder}/vq_{vq_tag}_indices_correration.png")
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from transformers import BertModel, BertConfig

# Load the pre-trained VQ-VAE embeddings
data_folder = "Seq2Seq/"
root = f"{data_folder}v1_BERT/"
os.makedirs(root, exist_ok=True)
vq_embeddings_path = f"{data_folder}f4_noattn_vq_weights.npy"
vq_embeddings = np.load(vq_embeddings_path)
print(vq_embeddings.shape)
print(f"Load vq_embeddings done from {vq_embeddings_path}")

# Configure the BERT model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
D = 12
config = BertConfig.from_pretrained('bert-base-uncased')
config.hidden_size = D  # Match to the dimension of your input
model = BertModel(config)

class BertForSeq2SeqRegression(nn.Module):
    def __init__(self, bert_model):
        super(BertForSeq2SeqRegression, self).__init__()
        self.bert = bert_model
        self.regression_head = nn.Linear(bert_model.config.hidden_size, D)

    def forward(self, x):
        # Forward pass through BERT
        outputs = self.bert(inputs_embeds=x)
        # Extract the last hidden state (sequence output)
        sequence_output = outputs.last_hidden_state
        # Apply the regression head
        predictions = self.regression_head(sequence_output)
        return predictions

# Initialize the custom model
model = BertForSeq2SeqRegression(model).to(device)
ckpt_path = f"{root}best_model.pth"
model.load_state_dict(torch.load(ckpt_path))
print(f"Load model from {ckpt_path}")


# Split the data into training, validation, and testing sets

# tag_list = [
#     "E4055", "E4058", "E4061",          "E4066",
#     "E4068", "E4069", "E4073", "E4074", "E4077",
#     "E4078", "E4079",          "E4081", "E4084",
#              "E4091", "E4092", "E4094", "E4096",
#              "E4098", "E4099",          "E4103",
#     "E4105", "E4106", "E4114", "E4115", "E4118",
#     "E4120", "E4124", "E4125", "E4128", "E4129",
#     "E4130", "E4131", "E4134", "E4137", "E4138",
#     "E4139",
# ]

test_tag_list = ["E4055"]

# randomly shuffle tag_list
# np.random.shuffle(tag_list)

# train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

# train_list = tag_list[:int(len(tag_list) * train_ratio)]
# val_list = tag_list[int(len(tag_list) * train_ratio):int(len(tag_list) * (train_ratio + val_ratio))]
# test_list = tag_list[int(len(tag_list) * (train_ratio + val_ratio)):]

def enumerate_cubes(data, cube_size):
    ax, ay, az = data.shape
    # pad the data if the size is not divisible by cube_size
    px = (cube_size - ax % cube_size) % cube_size
    py = (cube_size - ay % cube_size) % cube_size
    pz = (cube_size - az % cube_size) % cube_size
    data = np.pad(data, ((0, px), (0, py), (0, pz)), mode='constant')
    print(f"Pad the data to {data.shape} from ({ax}, {ay}, {az})")
    cubes = []
    for x in range(0, data.shape[0], cube_size):
        for y in range(0, data.shape[1], cube_size):
            for z in range(0, data.shape[2], cube_size):
                cube = data[x:x + cube_size, y:y + cube_size, z:z + cube_size]
                cubes.append(cube)
    return np.asarray(cubes)

def assemble_cube_array_from_seq_to_volume(cube_array, cube_size, data_shape):
    volume = np.zeros(data_shape)
    ax, ay, az = data_shape
    # pad the data if the size is not divisible by cube_size
    px = (cube_size - ax % cube_size) % cube_size
    py = (cube_size - ay % cube_size) % cube_size
    pz = (cube_size - az % cube_size) % cube_size
    volume = np.pad(volume, ((0, px), (0, py), (0, pz)), mode='constant')

    # count = np.zeros(data_shape)
    for idx, cube in enumerate(cube_array):
        x = idx // (data_shape[1] // cube_size)
        y = (idx // (data_shape[2] // cube_size)) % (data_shape[1] // cube_size)
        z = idx % (data_shape[2] // cube_size)
        volume[x * cube_size:(x + 1) * cube_size, y * cube_size:(y + 1) * cube_size, z * cube_size:(z + 1) * cube_size] += cube
        # count[x * cube_size:(x + 1) * cube_size, y * cube_size:(y + 1) * cube_size, z * cube_size:(z + 1) * cube_size] += 1

    # cut the padded part
    volume = volume[:ax, :ay, :az]

    return volume

# Start evaluating the model

optimizer = AdamW(model.parameters(), lr=5e-5)
# epochs = 1000
batch_size = 24
samples_per_case = 25
cube_size = 8
vq_embeddings = torch.tensor(vq_embeddings, dtype=torch.float32)
# val_per_epoch = 25
# save_per_epoch = 100

# best_val=1e6
# best_epoch=0

test_loss = []

for idx_tag, tag in enumerate(test_tag_list):
    STEP1_path = f"{data_folder}token_volume/TOKEN_STEP1_{tag}_VOLUME.npy"
    STEP2_path = f"{data_folder}token_volume/TOKEN_STEP2_{tag}_VOLUME.npy"
    STEP1_data = np.load(STEP1_path).reshape(-1, 100, 100)
    STEP2_data = np.load(STEP2_path).reshape(-1, 100, 100)
    case_loss = []
    STEP1_cube_array = enumerate_cubes(STEP1_data, cube_size)
    pred_STEP2_cube_array = []
    len_cubes = len(STEP1_cube_array)
    print(f"Enumerate {len_cubes} cubes")
    for idx_cube, STEP1_cube in enumerate(STEP1_cube_array):
        STEP1_seq = STEP1_cube.reshape(-1)
        token1 = vq_embeddings[STEP1_seq]
        token1 = token1.repeat(1,4)
        token1 = token1.unsqueeze(0).to(device)
        pred_STEP2_cube = model(token1)

        pred_1 = pred_STEP2_cube[:, :, :3].reshape(-1, 3).detach().cpu()
        pred_2 = pred_STEP2_cube[:, :, 3:6].reshape(-1, 3).detach().cpu()
        pred_3 = pred_STEP2_cube[:, :, 6:9].reshape(-1, 3).detach().cpu()
        pred_4 = pred_STEP2_cube[:, :, 9:12].reshape(-1, 3).detach().cpu()

        pred_idx_1 = torch.argmin(torch.cdist(pred_1, vq_embeddings), dim=1)
        pred_idx_2 = torch.argmin(torch.cdist(pred_2, vq_embeddings), dim=1)
        pred_idx_3 = torch.argmin(torch.cdist(pred_3, vq_embeddings), dim=1)
        pred_idx_4 = torch.argmin(torch.cdist(pred_4, vq_embeddings), dim=1)

        # select the majority vote from the 4 predictions
        pred_idx = torch.mode(torch.stack([pred_idx_1, pred_idx_2, pred_idx_3, pred_idx_4]), dim=0).values
        pred_STEP2_cube = pred_idx.reshape(cube_size, cube_size, cube_size) 
        pred_STEP2_cube_array.append(pred_STEP2_cube)
    
    # save the predicted STEP cube array
    savename = f"{data_folder}token_volume/PRED_STEP2_{tag}_CUBE_ARRAY.npy"
    np.save(savename, np.asarray(pred_STEP2_cube_array))
    print(f"Save the predicted STEP2 cube array to {savename}")
    
    pred_STEP2_data = assemble_cube_array_from_seq_to_volume(pred_STEP2_cube_array, cube_size, STEP2_data.shape)
    print(f"Predicted STEP2 data shape: {pred_STEP2_data.shape}")
    pred_STEP2_path = f"{data_folder}token_volume/PRED_STEP2_{tag}_VOLUME.npy"
    np.save(pred_STEP2_path, pred_STEP2_data)
    print(f"Save the predicted STEP2 data to {pred_STEP2_path}")

    # compute how many indices are different
    diff = np.sum(STEP2_data != pred_STEP2_data)
    print(f"Test Batch {idx_tag + 1}/{len(test_tag_list)}, Diff: {diff}, Percentage: {diff / STEP2_data.size:.4f}")

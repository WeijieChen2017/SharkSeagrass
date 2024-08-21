import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnected(nn.Module):
  def __init__(self,
               input_dim : int,
               hidden_dims: list,
               output_dim: int):
    super(FullyConnected, self).__init__()

    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.output_dim = output_dim

    self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
    self.layers = nn.ModuleList()
    # self.layers.append(nn.ReLU())
    for i in range(1, len(self.hidden_dims)):
      self.layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
      self.layers.append(nn.ReLU())
    self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

    self.layers = nn.Sequential(*self.layers)
    self.init_weights()

  def init_weights(self):
    # initialize the weights of the layers
    for layer in self.layers:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    # init self.input_layer
    nn.init.xavier_uniform_(self.input_layer.weight)
    nn.init.zeros_(self.input_layer.bias)

  def forward(self, x):
    res = self.input_layer(x)
    for layer in self.layers:
      res = layer(res)
    return x + res

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

import os
import json
import torch
import random
import numpy as np

# basic setting
VQ_NAME = "f4"
VQ_FACTOR = 4
root_folder = f"./B100/vq_{VQ_NAME}_FCN"

# import tags from the json file
train_tags = json.load(open(root_folder+f"/train_tags.json", "r"))
val_tags = json.load(open(root_folder+f"/val_tags.json", "r"))
test_tags = json.load(open(root_folder+f"/test_tags.json", "r"))

# with open(root_folder+f"/train_tags.json", "w") as f:
#     json.dump(train_tags, f)
# with open(root_folder+f"/val_tags.json", "w") as f:
#     json.dump(val_tags, f)
# with open(root_folder+f"/test_tags.json", "w") as f:
#     json.dump(test_tags, f)

# load the data
# train_PET_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_PET_ind.npy" for name_tag in train_tags]
# train_CTr_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_CTr_ind.npy" for name_tag in train_tags]

# val_PET_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_PET_ind.npy" for name_tag in val_tags]
# val_CTr_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_CTr_ind.npy" for name_tag in val_tags]

test_PET_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_PET_ind.npy" for name_tag in test_tags]
test_CTr_list = [f"./B100/vq_{VQ_NAME}_ind/vq_{VQ_NAME}_{name_tag}_CTr_ind.npy" for name_tag in test_tags]

# train_dataset = []
# val_dataset = []
test_dataset = []

# for PET_path, CTr_path in zip(train_PET_list, train_CTr_list):
#     PET = np.load(PET_path)
#     CTr = np.load(CTr_path)
#     train_dataset.append({"PET_data": PET, "CTr_data": CTr})

# for PET_path, CTr_path in zip(val_PET_list, val_CTr_list):
#     PET = np.load(PET_path)
#     CTr = np.load(CTr_path)
#     val_dataset.append({"PET_data": PET, "CTr_data": CTr})

for PET_path, CTr_path in zip(test_PET_list, test_CTr_list):
    PET = np.load(PET_path)
    CTr = np.load(CTr_path)
    test_dataset.append({"PET_data": PET, "CTr_data": CTr, "PET_path": PET_path, "CTr_path": CTr_path})

# print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

print(f"The testing dataset has {len(test_dataset)} samples.")
for tag in test_tags:
    print(tag)

# load the VQ codebook
model_state_dict = torch.load(f'vq_{VQ_NAME}.ckpt')['state_dict']
vq_weights = model_state_dict['quantize.embedding.weight']
vq_weights_cpu = vq_weights.cpu().detach().numpy()
n_embed_dim = vq_weights.shape[1]
print("VQ weights shape:", vq_weights.shape)
criterion = nn.MSELoss()

# build the model
input_dim = (400 // VQ_FACTOR) * (400 // VQ_FACTOR) * n_embed_dim
hidden_dims = [input_dim // 2, input_dim // 2]
output_dim = input_dim

model_params = {
    "input_dim" : input_dim,
    "hidden_dims" : hidden_dims,
    "output_dim" : output_dim
}

model_embed = FullyConnected(**model_params).to("cuda")
state_dict_path = f"best_model_at_all_h{hidden_dims}.pth"
model_embed.load_state_dict(torch.load(state_dict_path))

# testing
model_embed.eval()
overall_test_loss = 0.0
overall_embedding_mismatch_count = 0

for test_dict in test_dataset:
    test_loss = 0.0
    embedding_mismatch_count = 0
    input_full = test_dict["PET_data"]
    output_full = test_dict["CTr_data"]
    len_z = input_full.shape[0]
    pred_full = np.zeros_like(output_full)
    for i in range(len_z):
        input_embed = vq_weights[input_full[i]]
        output_embed = vq_weights[output_full[i]]
        input_embed = input_embed.reshape(1, -1)
        output_embed = output_embed.reshape(1, -1)

        with torch.no_grad():
            pred_embed = model_embed(input_embed)
            loss = criterion(pred_embed, output_embed)
            test_loss += loss.item()
        
        pred_embed = np.squeeze(pred_embed.detach().cpu().numpy()) # (1, 10000)

        correct_indices = input_full[i] # Shape: (10000,)
        pred_indices = np.reshape(pred_embed, (1, 10000, n_embed_dim)) # Shape: (1, 10000, n_embed_dim)
        print("Pred embed shape:", pred_embed.shape)
        pred_indices = np.argmin(np.linalg.norm(vq_weights_cpu - pred_indices, axis=2), axis=1) # Shape: (10000,)
        pred_full[i] = pred_indices
        # count how many indices are mismatched
        mismatch_count = np.sum(correct_indices != pred_indices)
        embedding_mismatch_count += mismatch_count
        print(f"Index z = {i}, embedding Mismatch Count: {mismatch_count} loss: {loss.item()}")

    test_loss = test_loss / len(test_dataset) / len_z / n_embed_dim
    print(f"Test Loss: {test_loss:.6e}")
    print(f"Embedding Mismatch Count: {embedding_mismatch_count // len(test_dataset)}")

    overall_test_loss += test_loss
    overall_embedding_mismatch_count += embedding_mismatch_count

overall_test_loss = overall_test_loss / len(test_dataset)
overall_embedding_mismatch_count = overall_embedding_mismatch_count // len(test_dataset)
print(f"Overall Test Loss: {overall_test_loss:.6e}")
print(f"Overall Embedding Mismatch Count: {overall_embedding_mismatch_count}")

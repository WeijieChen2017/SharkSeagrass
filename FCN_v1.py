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
new_folders = [
    root_folder,
]
for folder in new_folders:
    os.makedirs(folder, exist_ok=True)

# randomly split the data
random.seed(42)
random.shuffle(tag_list)

train_tags = tag_list[:int(0.8*len(tag_list))]
val_tags = tag_list[int(0.8*len(tag_list)):0.9*len(tag_list)]
test_tags = tag_list[int(0.9*len(tag_list)):]

log_filename = f"./B100/vq_{VQ_NAME}_FCN/log.txt"
with open(log_filename, "w") as f:
    f.write("")
    f.write(f"Train tags: {train_tags}\n")
    f.write(f"Val tags: {val_tags}\n")
    f.write(f"Test tags: {test_tags}\n")

# save the division into json
with open(root_folder+f"/train_tags.json", "w") as f:
    json.dump(train_tags, f)
with open(root_folder+f"/val_tags.json", "w") as f:
    json.dump(val_tags, f)
with open(root_folder+f"/test_tags.json", "w") as f:
    json.dump(test_tags, f)

# load the data
train_PET_list = [f"./B100/npy/PET_TOFNAC_{name_tag}.npy" for name_tag in train_tags]
train_CTr_list = [f"./B100/npy/CTACIVV_{name_tag}.npy" for name_tag in train_tags]

val_PET_list = [f"./B100/npy/PET_TOFNAC_{name_tag}.npy" for name_tag in val_tags]
val_CTr_list = [f"./B100/npy/CTACIVV_{name_tag}.npy" for name_tag in val_tags]

test_PET_list = [f"./B100/npy/PET_TOFNAC_{name_tag}.npy" for name_tag in test_tags]
test_CTr_list = [f"./B100/npy/CTACIVV_{name_tag}.npy" for name_tag in test_tags]

train_dataset = []
val_dataset = []
test_dataset = []

for PET_path, CTr_path in zip(train_PET_list, train_CTr_list):
    PET = np.load(PET_path)
    CTr = np.load(CTr_path)
    train_dataset.append({"PET_data": PET, "CTr_data": CTr})

for PET_path, CTr_path in zip(val_PET_list, val_CTr_list):
    PET = np.load(PET_path)
    CTr = np.load(CTr_path)
    val_dataset.append({"PET_data": PET, "CTr_data": CTr})

for PET_path, CTr_path in zip(test_PET_list, test_CTr_list):
    PET = np.load(PET_path)
    CTr = np.load(CTr_path)
    test_dataset.append({"PET_data": PET, "CTr_data": CTr})

print(len(train_dataset), len(val_dataset), len(test_dataset))


# load the VQ codebook
model_state_dict = torch.load(f'vq_{VQ_NAME}.ckpt')['state_dict']
vq_weights = model_state_dict['quantize.embedding.weight']
print("VQ weights shape:", vq_weights.shape)

# build the model
input_dim = (400 // VQ_FACTOR) * (400 // VQ_FACTOR) * VQ_FACTOR
hidden_dims = [input_dim // 2, input_dim // 2]
output_dim = input_dim

model_params = {
    "input_dim" : input_dim,
    "hidden_dims" : hidden_dims,
    "output_dim" : output_dim
}

model = FullyConnected(**model_params).to("cuda")
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.09)
criterion = nn.MSELoss()

# training setting
num_epoch = 1000
batch_size = 64
best_eval_loss = float("inf")
n_embed_dim = 4
n_output_epoch = 100

for epoch in range(num_epoch):

    # train
    model.train()
    train_loss = 0.0
    random.shuffle(train_dataset)
    for train_dict in train_dataset:
        input_full = train_dict["PET_data"]
        output_full = train_dict["CTr_data"]
        print("Input shape:", input_full.shape, output_full.shape)
        exit()
        train_loss = 0.0

        for i in range(0, len(input_full), batch_size):
            if i + batch_size > len(input_full):
                break
            else:
                input_batch = input_full[i:i+batch_size, :, :]
                output_batch = output_full[i:i+batch_size, :, :]

                # flatten batch at 3rd dim
                input_batch = input_batch.reshape(batch_size, -1)
                output_batch = output_batch.reshape(batch_size, -1)
                # print("Index i:", i, input_batch.shape, output_batch.shape)

                input_embed = torch.tensor(input_batch).float().to("cuda")
                output_embed = torch.tensor(output_batch).float().to("cuda")

                optimizer.zero_grad()
                pred_embed = model(input_embed)
                loss = criterion(pred_embed, output_embed)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

    train_loss = train_loss / n_batch / batch_size / len(train_tags)
    if epoch % n_output_epoch == 0:
        print(f"Epoch: [{epoch}]/[{num_epoch}], Loss: {train_loss:.6e}", end=" <> ")

    # eval
    model.eval()
    eval_loss = 0.0
    for tag in val_tags:
        input_full = dataset[tag]["PET"]
        output_full = dataset[tag]["CTr"]
        random.shuffle(input_full)
        random.shuffle(output_full)
        n_batch = len(input_full) // batch_size


        for i in range(0, len(input_full), batch_size):
            if i + batch_size > len(input_full):
                break
            else:
                input_batch = input_full[i:i+batch_size, :, :]
                output_batch = output_full[i:i+batch_size, :, :]

                # flatten batch at 3rd dim
                input_batch = input_batch.reshape(batch_size, -1)
                output_batch = output_batch.reshape(batch_size, -1)

                input_embed = torch.tensor(input_batch).float().to("cuda")
                output_embed = torch.tensor(output_batch).float().to("cuda")

                with torch.no_grad():
                    pred_embed = model(input_embed)
                    loss = criterion(pred_embed, output_embed)
                    eval_loss += loss.item()

    eval_loss = eval_loss / n_batch / batch_size / len(val_tags)
    if epoch % n_output_epoch == 0:
        print(f"Eval Loss: {eval_loss:.6e}")

    # save the best model
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f"best_model_at_all_h{hidden_dims}.pth")
        print(f"Saved best model at epoch {best_epoch} with eval loss {eval_loss:.6e}")

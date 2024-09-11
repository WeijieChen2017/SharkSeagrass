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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Split the data into training, validation, and testing sets

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

# randomly shuffle tag_list
np.random.shuffle(tag_list)

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

train_list = tag_list[:int(len(tag_list) * train_ratio)]
val_list = tag_list[int(len(tag_list) * train_ratio):int(len(tag_list) * (train_ratio + val_ratio))]
test_list = tag_list[int(len(tag_list) * (train_ratio + val_ratio)):]

def draw_cube(STEP1_data, STEP2_data, cube_size, batch_size):

    batch_x = []
    batch_y = []

    for idx_batch in range(batch_size):
        cx = np.random.randint(cube_size // 2, STEP1_data.shape[1] - cube_size // 2)
        cy = np.random.randint(cube_size // 2, STEP1_data.shape[2] - cube_size // 2)
        cz = np.random.randint(cube_size // 2, STEP1_data.shape[0] - cube_size // 2)
        step1_cube = STEP1_data[cz - cube_size // 2:cz + cube_size // 2, cx - cube_size // 2:cx + cube_size // 2, cy - cube_size // 2:cy + cube_size // 2]
        step2_cube = STEP2_data[cz - cube_size // 2:cz + cube_size // 2, cx - cube_size // 2:cx + cube_size // 2, cy - cube_size // 2:cy + cube_size // 2]
        step1_seq = step1_cube.reshape(-1)
        step2_seq = step2_cube.reshape(-1)
        token1 = vq_embeddings[step1_seq]
        token2 = vq_embeddings[step2_seq]
        token1 = token1.repeat(1,4)
        token2 = token2.repeat(1,4)
        token1 = token1.unsqueeze(0)
        token2 = token2.unsqueeze(0)

        batch_x.append(token1)
        batch_y.append(token2)

    batch_x = torch.cat(batch_x, dim=0)
    batch_y = torch.cat(batch_y, dim=0)

    return batch_x, batch_y


# Start training

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 1000
batch_size = 48
samples_per_case = 20
cube_size = 8
vq_embeddings = torch.tensor(vq_embeddings, dtype=torch.float32)
val_per_epoch = 50
save_per_epoch = 100

best_val=1e6
best_epoch=0

for epoch in range(epochs):

    # train
    model.train()
    random.shuffle(train_list)
    epoch_loss = []
    for idx_tag, tag in enumerate(train_list):
        STEP1_path = f"{data_folder}token_volume/TOKEN_STEP1_{tag}_VOLUME.npy"
        STEP2_path = f"{data_folder}token_volume/TOKEN_STEP2_{tag}_VOLUME.npy"
        STEP1_data = np.load(STEP1_path).reshape(-1, 100, 100)
        STEP2_data = np.load(STEP2_path).reshape(-1, 100, 100)
        case_loss = []
        for idx_sample in range(samples_per_case):
            batch_x, batch_y = draw_cube(STEP1_data, STEP2_data, cube_size, batch_size)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = F.mse_loss(predictions, batch_y)
            loss.backward()
            optimizer.step()
            case_loss.append(loss.item())
            # print(f'Training Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(train_list)}, Sample {idx_sample + 1}/{samples_per_case}, Loss: {loss.item():.4f}')

        case_loss = np.mean(np.asarray(case_loss))
        epoch_loss.append(case_loss)
        print(f'Training Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(train_list)}, Loss: {case_loss:.4f}')

    epoch_loss = np.mean(np.asarray(epoch_loss))
    print(f'Training Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    # eval
    if epoch % val_per_epoch == 0:
        model.eval()
        random.shuffle(val_list)
        val_loss = []
        for idx_tag, tag in enumerate(val_list):
            STEP1_path = f"{data_folder}token_volume/TOKEN_STEP1_{tag}_VOLUME.npy"
            STEP2_path = f"{data_folder}token_volume/TOKEN_STEP2_{tag}_VOLUME.npy"
            STEP1_data = np.load(STEP1_path).reshape(-1, 100, 100)
            STEP2_data = np.load(STEP2_path).reshape(-1, 100, 100)
            case_loss = []
            for idx_sample in range(samples_per_case):
                batch_x, batch_y = draw_cube(STEP1_data, STEP2_data, cube_size, batch_size)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = F.mse_loss(predictions, batch_y)
                case_loss.append(loss.item())
                # print(f'Validation Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(val_list)}, Sample {idx_sample + 1}/{samples_per_case}, Loss: {loss.item():.4f}')

            case_loss = np.mean(np.asarray(case_loss))
            val_loss.append(case_loss)
            print(f'Validation Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(val_list)}, Loss: {case_loss:.4f}')
        
        val_loss = np.mean(np.asarray(val_loss))
        print(f'Validation Epoch {epoch + 1}/{epochs}, Loss: {val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_path = f"{root}best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Save model at epoch {epoch + 1}")

        # Test the model
        model.eval()
        random.shuffle(test_list)
        test_loss = []
        for idx_tag, tag in enumerate(test_list):
            STEP1_path = f"{data_folder}token_volume/TOKEN_STEP1_{tag}_VOLUME.npy"
            STEP2_path = f"{data_folder}token_volume/TOKEN_STEP2_{tag}_VOLUME.npy"
            STEP1_data = np.load(STEP1_path).reshape(-1, 100, 100)
            STEP2_data = np.load(STEP2_path).reshape(-1, 100, 100)
            case_loss = []
            for idx_sample in range(samples_per_case):
                batch_x, batch_y = draw_cube(STEP1_data, STEP2_data, cube_size, batch_size)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions = model(batch_x)
                loss = F.mse_loss(predictions, batch_y)
                case_loss.append(loss.item())
                # print(f'Test Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(test_list)}, Sample {idx_sample + 1}/{samples_per_case}, Loss: {loss.item():.4f}')

            case_loss = np.mean(np.asarray(case_loss))
            test_loss.append(case_loss)
            print(f'Test Epoch {epoch + 1}/{epochs}, Batch {idx_tag + 1}/{len(test_list)}, Loss: {case_loss:.4f}')
    
    if epoch % save_per_epoch == 0:
        save_path = f"{root}model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Save model at epoch {epoch + 1}")
            


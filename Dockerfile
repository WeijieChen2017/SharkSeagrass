# Use the MONAI base image
# FROM projectmonai/monai:latest
FROM huggingface/transformers-pytorch-gpu

# Install additional Python packages using pip
RUN pip install wandb
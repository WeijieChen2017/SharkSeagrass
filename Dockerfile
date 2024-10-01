# Use the MONAI base image
# FROM projectmonai/monai:latest
FROM pull huggingface/transformers-pytorch-gpu

# Install additional Python packages using pip
RUN pip install wandb
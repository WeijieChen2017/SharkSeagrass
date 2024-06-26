# Use the MONAI base image
FROM projectmonai/monai:latest

# Install additional Python packages using pip
RUN pip install omegaconf wandb
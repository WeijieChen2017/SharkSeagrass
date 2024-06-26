# Use the MONAI base image
FROM projectmonai/monai:latest

# Set environment variables for Python and CUDA
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages using pip
RUN pip install --no-cache-dir \
    torch==1.13.0+cu113 \
    torchvision==0.14.0 \
    torchtext==0.11.0 \
    torch-tensorrt==1.2.0a0 \
    einops \
    pytorch-lightning==1.7.7 \
    omegaconf==2.1.1 \
    protobuf==3.20.1

# Workaround for SSL certificate verification issues
RUN apt-get install -y ca-certificates
RUN update-ca-certificates --fresh

# Copy your code into the Docker image
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Print CUDA device info
RUN python -c "import torch; print('The device is:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"

# Set entrypoint or CMD to run your training script or other commands
# ENTRYPOINT ["python", "your_script.py"]

# Use MONAI's latest image as the base image
FROM projectmonai/monai:latest

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /usr/local/bin:$PATH

# Install necessary dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libcurl4-openssl-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    ftfy==6.1.1 \
    lpips==0.1.4 \
    regex==2021.10.8 \
    pytorch-lightning==1.5.10 \
    einops==0.3.0 \
    omegaconf==2.0.0 \
    lmdb==1.0.0 \
    wandb==0.12.21 \
    git+https://github.com/openai/CLIP.git \
    albumentations==0.4.3 \
    kornia==0.5.11 \
    Pillow==9.0.1

# Optionally, specify the versions for torch and torchvision if needed
# RUN pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory
WORKDIR /workspace

# Define the default command
CMD ["bash"]
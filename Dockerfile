# Use a stable CUDA base image
FROM nvidia/cuda:11.0-base-ubuntu20.04

# Set environment variables to prevent interactive prompts during the build
ENV DEBIAN_FRONTEND=noninteractive

# Disable problematic NVIDIA repositories (if any)
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Install Python and basic dependencies
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    python -m pip install --upgrade pip==20.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible PyTorch
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install Python packages specified in the environment.yaml file
RUN pip install numpy==1.19.2 albumentations==0.4.3 opencv-python==4.1.2.30 pudb==2019.2 \
    imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 \
    einops==0.3.0 torch-fidelity==0.3.0 transformers==4.3.1 streamlit>=0.73.1 test-tube>=0.7.5

# Install additional GitHub repositories with editable mode
RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers && \
    pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip && \
    pip install -e .

# Set default command to open a shell
CMD ["/bin/bash"]

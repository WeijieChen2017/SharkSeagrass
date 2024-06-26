# Use the MONAI base image
FROM projectmonai/monai:latest

# Set environment variables for Python
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
RUN pip install --no-cache-dir --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user \
    torch \
    numpy \
    einops \
    pytorch-lightning \
    omegaconf

# Copy your code into the Docker image
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Print CUDA device info
RUN python -c "import torch; print('The device is:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"

# Set entrypoint or CMD to run your training script or other commands
# ENTRYPOINT ["python", "your_script.py"]

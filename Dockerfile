# Use MONAI's latest image as the base image
FROM projectmonai/monai:latest

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

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

# Copy the environment.yaml file
COPY environment.yaml /tmp/environment.yaml

# Create the conda environment
RUN conda env create -f /tmp/environment.yaml && conda clean -a

# Set the default environment
RUN echo "source activate enhancing" > ~/.bashrc
ENV PATH /opt/conda/envs/enhancing/bin:$PATH

# Install any additional packages
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Set the working directory
WORKDIR /workspace

# Define the default command
CMD ["bash"]

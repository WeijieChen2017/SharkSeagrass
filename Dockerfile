# Use a base image with CUDA support
FROM projectmonai/monai:latest

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

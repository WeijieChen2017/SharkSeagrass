# SharkSeagrass

SharkSeagrass is a machine learning project dedicated to synthesizing CT images from PET scans using advanced Vector Quantization (VQ) encoder-decoder frameworks. This project leverages large-scale public datasets and innovative model architectures to improve synthetic CT quality, especially with limited paired PET-CT data.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Experiments](#experiments)
- [Evaluation Metrics](#evaluation-metrics)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The main objective of SharkSeagrass is to synthesize high-quality CT images from PET images. The research explores different model architectures, including convolutional and transformer-based encoder-decoders, to achieve this goal. The project also investigates techniques like VQ-GAN and pyramid image generation to enhance image detail and quality.

## Datasets
### Public Dataset
- **AbdomenAtlas-8K**: 8,448 CT volumes with per-voxel annotated abdominal organs.
- Other datasets for comparison: AMOS, TotalSegmentator, AbdomenCT-1K.

### Private Dataset
- **PET-CT Paired Dataset**: Registered PET-CT 3D whole-body dataset.

## Model Architecture
The primary framework is a VQ encoder-decoder with the following components:
- **Convolutional Models**: ResNet, UNet, DenseNet for feature extraction and image generation.
- **Transformer Models**: Vision Transformer (ViT), Swin Transformer for larger pre-trained datasets.

## Training Procedure
### Stage 1: Self-Supervised Training
- Train the VQ encoder-decoder using the AbdomenAtlas-8K dataset.

### Stage 2: Supervised Training
- Train using the paired PET-CT dataset to align PET embeddings with CT embeddings.

## Experiments
1. **Convolutional Encoder-Decoder**
2. **Transformer Encoder-Decoder**
3. **Enhancing Image Details**: VQ-GAN and pyramid image generation techniques.

## Evaluation Metrics
- **Quantitative**: PSNR, SSIM, MSE.
- **Qualitative**: Visual inspection by radiologists, comparison with ground truth CT images.

## Getting Started
### Prerequisites
- Python 3.7+
- PyTorch or TensorFlow
- High-performance computing resources (GPUs)

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/SharkSeagrass.git
cd SharkSeagrass
```

### References

1. **Fundamental VQVAE PyTorch Implementation**:
   - Repository: https://github.com/MishaLaskin/vqvae
   - Description: This repository provides the fundamental implementation of VQVAE using PyTorch. It serves as a starting point for CNN-based VQVAE models.

2. **VQ-Specific Folder**:
   - Repository: https://github.com/lucidrains/vector-quantize-pytorch
   - Description: This folder contains VQ-specific code, which can be referenced for further understanding.

3. **Variety of VQVAE Mutants**:
   - Repository: https://github.com/rese1f/Awesome-VQVAE
   - Description: Explore this repository for different VQVAE variants and mutants.

4. **VQVAE and VQGAN with CNN-Based Improvements**:
   - Repository: https://github.com/CompVis/taming-transformers
   - Description: This repository includes implementations of VQVAE and VQGAN, along with a list of enhancements based on convolutions.

5. **Transformer-Based ViT-VQGAN Implementation**:
   - Repository: https://github.com/thuanz123/enhancing-transformers
   - Description: An unofficial implementation of ViT-VQGAN in PyTorch.
# MSAI 437 Deep Learning Final Project - Simple DDIM from scratch

This repository contains the implementation of a simple Denoising Diffusion Implicit Models (DDIM) for the MSAI 437 Deep Learning course at Northwestern University. The project aims to demonstrate the application of diffusion models in generating high-quality samples from a data distribution.

## Overview

Denoising Diffusion Implicit Models (DDIM) are a class of generative models that iteratively convert noise into samples from the target distribution. This project implements a simplified version of DDIM to understand the underlying principles and to explore its capabilities in generating complex data distributions.

## Model outputs
![Comparison](https://github.com/yashkhurana24/ddim/blob/main/model_outptus/comparison.png)

- Dataset can be found [here](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- Download trained model weights from [here](https://drive.google.com/drive/folders/1URqy-GJMKjxyuAwFfvdAhOVrT2dGvz9o?usp=sharing) and place in ./trained_models

## Installation

To set up the project environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yashkhurana24/ddim
cd ddim

# Create a virtual environment (optional)
conda create -n ddim python=3.8
conda activate ddim

# Install the required dependencies
# Install PyTorch from the official website (compiled with CUDA)
pip install -r requirements.txt
```

### To-do
- ~~training~~
- ~~add attention mechanism to unet architecture~~
- ~~evaluation of trained model~~
- ~~inference from trained model~~
- ~~add trained models and model outputs~~
- ~~train GAN on same data for performance comparison~~
- remove all dependencies except pytorch
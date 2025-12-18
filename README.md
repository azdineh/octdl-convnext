# OCTDL Classification with ConvNeXt

This repository contains the official implementation for training a ConvNeXt-Tiny model on the OCTDL (Optical Coherence Tomography) dataset. It classifies OCT images into 7 categories: AMD, DME, ERM, NO, RAO, RVO, and VID.

## Project Structure

* `config.py`: hyperparameters and file paths.
* `dataset.py`: Custom PyTorch Dataset and transformations.
* `train.py`: Main script for training and validation.
* `utils.py`: Utilities for evaluation and Grad-CAM visualization.

## Requirements

* Python 3.11+
* PyTorch
* timm
* scikit-learn
* grad-cam

Install dependencies via:
```bash
pip install -r requirements.txt

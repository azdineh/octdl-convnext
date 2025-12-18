import torch
from pathlib import Path

# Paths
# CHANGE THIS PATH to point to your local dataset
DATA_ROOT = Path('/kaggle/input/octdl-optical-coherence-tomography-dataset/OCTDL/OCTDL') 
OUTPUT_DIR = Path('.')

# Model Settings
MODEL_NAME = 'convnext_tiny'
IMAGE_SIZE = 224
NUM_CLASSES = 7  # AMD, DME, ERM, NO, RAO, RVO, VID
PRETRAINED = True

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
SEED = 42
NUM_WORKERS = 2

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

def get_data_splits():
    """Parses the dataset directory and creates train/val/test splits."""
    random.seed(config.SEED)
    np.random.seed(config.SEED)

    # 1. Detect Classes
    classes = [p.name for p in config.DATA_ROOT.iterdir() if p.is_dir()]
    classes = sorted(classes)
    print(f"Classes found: {classes}")

    # 2. Build file list
    items = []
    for idx, cls in enumerate(classes):
        class_dir = config.DATA_ROOT / cls
        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', 'bmp', 'tif', 'tiff']:
                items.append((str(img_path), idx, cls))
    
    df = pd.DataFrame(items, columns=['path', 'label', 'class'])
    
    # 3. Split (Stratified)
    # 20% Test, then 15% of remaining 80% for Validation
    train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=config.SEED)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, stratify=train_val_df['label'], random_state=config.SEED)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Dataset Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, classes

class OCTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])
        return img, label

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transforms, val_transforms

def get_dataloaders():
    train_df, val_df, test_df, classes = get_data_splits()
    train_tfm, val_tfm = get_transforms()

    train_ds = OCTDataset(train_df, transform=train_tfm)
    val_ds = OCTDataset(val_df, transform=val_tfm)
    test_ds = OCTDataset(test_df, transform=val_tfm)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader, classes

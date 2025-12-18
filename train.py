import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchmetrics.classification import MulticlassAccuracy

import config
from dataset import get_dataloaders
from utils import plot_confusion_matrix, print_metrics

def train_model():
    # 1. Setup Data
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    # 2. Setup Model
    print(f"Creating model: {config.MODEL_NAME}")
    model = timm.create_model(config.MODEL_NAME, pretrained=config.PRETRAINED, num_classes=len(classes))
    model.to(config.DEVICE)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # 4. Training Loop
    best_val_acc = 0.0
    best_model_path = config.OUTPUT_DIR / 'best_model.pth'

    print("Starting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        start_time = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'classes': classes,
                'val_acc': val_acc
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")

    print("Training complete.")
    
    # 5. Final Test
    print("Loading best model for testing...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Final Test Result -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Generate Confusion Matrix and Detailed Report
    print_metrics(model, test_loader, classes, config.DEVICE)

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss_sum += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
    return loss_sum / total, correct / total

if __name__ == "__main__":
    train_model()

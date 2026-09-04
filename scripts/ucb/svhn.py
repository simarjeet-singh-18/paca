#!/usr/bin/env python3
"""
Vision Transformer (ViT) Fine-tuning Script for Caltech-101 Dataset
"""

import os
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel
import os, time

# Configuration
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 4

# Paths
DATA_DIR = "../../data/SVHN"
SAVE_DIR = "../../saved_models_pretrained/vit_svhn"
os.makedirs(SAVE_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_transforms():
    """Define image transformations for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders():
    """Create training and validation dataloaders"""
    train_transform, val_transform = get_transforms()
    
    # Download and load Flowers102 dataset directly from torchvision
    train_dataset = datasets.SVHN(
        root="./data", 
        split="train", 
        transform=train_transform, 
        download=True
    )
    
    val_dataset = datasets.SVHN(
        root="./data", 
        split="test", 
        transform=val_transform, 
        download=True
    )
    
    # Get number of classes (svhn has 10 classes)
    
    num_classes = 10
    
    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")
    print(f"Number of classes: {num_classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, num_classes

def create_model(num_classes):
    """Create and configure Vision Transformer model"""
    # Load pre-trained ViT-B/16 model
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Replace the classifier head for our number of classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Unfreeze the classifier head
    for param in model.heads.parameters():
        param.requires_grad = True
    
    
    return model




print("=" * 60)
print("Vision Transformer Fine-tuning on SVHN")
print("=" * 60)

# Set random seeds for reproducibility
random_seed = np.random.randint(0, 10000)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Create dataloaders
print("Loading dataset...")
trainloader, testloader, num_classes = create_dataloaders()

# Create model
print(f"\nCreating ViT model for {num_classes} classes...")
model_vanilla = create_model(num_classes)
model_vanilla = model_vanilla.to(device)



peft_config = PacaConfig(
    r=48,
    target_modules=[
        "out_proj",  # Attention output projections (12 layers)
        "mlp.0",     # MLP expansion layers (768->3072, 12 layers) 
        "mlp.3",      # MLP contraction layers (3072->768, 12 layers)
    ], 
    paca_alpha=96,
    bias="all",
    modules_to_save=["head"],
    task_type=TaskType.FEATURE_EXTRACTION,
    steps_per_epoch=len(trainloader),
    arm_distribution="horizontal",  # or "vertical"
    num_arms=3,
    num_arms_to_select=1,
    ucb_alpha=6,
)


print(f"PEFT config: {peft_config}")

model = PacaModel(model_vanilla, peft_config, adapter_name="default")

model.print_trainable_parameters()

# Model, Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine schedule with warmup
total_epochs = 1000
warmup_epochs = 10

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

def evaluate(model, dataloader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() / len(dataloader)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    val_acc = 100. * correct / total
    return val_loss, val_acc

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=7):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    total_time = 0.0
    prev_val_acc = None
    print("Calculating initial validation accuracy...")
    _, initial_val_acc = evaluate(model, val_loader, criterion)
    print(f"Initial Val Acc: {initial_val_acc:.2f}%")
    prev_val_acc = initial_val_acc

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(train_loader)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss = running_loss
        train_acc = 100. * train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        # Compute reward (jump in validation accuracy)
        if prev_val_acc is not None:
            reward = val_acc - prev_val_acc
            model.update_ucb_rewards("default", reward)
        prev_val_acc = val_acc
        model.reselect_paca_weights("default")


        epoch_time = time.time() - start_time
        total_time += epoch_time
        model.print_logs("default")

        print(f"\nEpoch {epoch+1} Summary | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)

        

        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                print(f"Total training time far: {total_time:.2f} seconds")
                print(f"Average time per epoch: {total_time / (epoch + 1):.2f} seconds")
                break

    best_indices = model.get_best_indices("default")
    torch.save(best_indices, "best_indices.pt")
    # Load best model before returning
    # if best_model_state:
    #     model.load_state_dict(best_model_state)
    return model, best_indices


print("Fine tuning vit using UCB-based PaCA:")




def second_training_run(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, best_indices, patience=7):
    print("Starting second training: ")
    # Load and set best indices
    # best_indices = torch.load("best_indices.pt")
    model.set_fixed_indices(best_indices, "default")
    
    torch.manual_seed(42)  # Ensure consistent arm_indices
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    total_time = 0.0
    print("Calculating initial validation accuracy...")
    _, initial_val_acc = evaluate(model, val_loader, criterion)
    print(f"Initial Val Acc: {initial_val_acc:.2f}%")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(train_loader)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss = running_loss
        train_acc = 100. * train_correct / train_total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time
        model.print_logs("default")

        print(f"\nEpoch {epoch+1} Summary | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)

        

        # Early Stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                print(f"Total training time far: {total_time:.2f} seconds")
                print(f"Average time per epoch: {total_time / (epoch + 1):.2f} seconds")
                break
    
    return model
    
    
    
    
model = model.to(device)
# Run pre-training

# First training run
model, best_indices = train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=total_epochs)

# Reset model_vanilla for second run
# Create model
print(f"\nCreating ViT model for {num_classes} classes...")
model_vanilla = create_model(num_classes)
model_vanilla = model_vanilla.to(device)


peft_config_fixed = PacaConfig(
    r=48,
    target_modules=[
        "out_proj",  # Attention output projections (12 layers)
        "mlp.0",     # MLP expansion layers (768->3072, 12 layers) 
        "mlp.3",      # MLP contraction layers (3072->768, 12 layers)
    ],
    paca_alpha=96,
    bias="all",
    modules_to_save=["heads.head"],
    steps_per_epoch=None,  # Disable UCB reselection
    arm_distribution="horizontal",
    num_arms=3,
    num_arms_to_select=1,
    ucb_alpha=6,
    use_fixed_indices=True,
)

model_fixed = PacaModel(model_vanilla, peft_config_fixed, adapter_name="default")

model_fixed.print_trainable_parameters()
model_fixed = model_fixed.to(device)

# Create new optimizer for second run
optimizer_fixed = AdamW(model_fixed.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_fixed = LambdaLR(optimizer_fixed, lr_lambda=lr_lambda)

# Run second training
model_fixed = second_training_run(model_fixed, trainloader, testloader, criterion, optimizer_fixed, scheduler_fixed, epochs=total_epochs, best_indices=best_indices)


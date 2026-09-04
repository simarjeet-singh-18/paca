#!/usr/bin/env python3
"""
Vision Transformer (ViT) Fine-tuning Script for Oxford-IIIT Pets with Gradient Chain Selection
Uses gradient-based chain construction to create neural pathways through the network.
Fixed 25 epochs training without early stopping.
"""
import os, time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from peft import get_peft_model, PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel

# Configuration
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 4

# Paths
DATA_DIR = "../../data/pets"
SAVE_DIR = "../../saved_models_pretrained/vit_pets_gradient_chain_fixed"
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

    # Download and load Oxford-IIIT Pet dataset
    train_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        transform=train_transform,
        download=True
    )

    val_dataset = datasets.OxfordIIITPet(
        root="./data",
        split="test",
        transform=val_transform,
        download=True
    )

    # Get number of classes (Oxford-IIIT Pet has 37 classes)
    num_classes = 37

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

    # Replace the classifier head
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # Unfreeze the classifier head
    for param in model.heads.parameters():
        param.requires_grad = True

    return model


print("=" * 60)
print("Vision Transformer Fine-tuning on Oxford-IIIT Pets")
print("Using Gradient Chain Selection - Fixed 25 Epochs")
print("=" * 60)

# Set random seeds
random_seed = np.random.randint(0, 10000)
print(f"Using random seed: {random_seed}")
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Create dataloaders
print("\nLoading dataset...")
trainloader, testloader, num_classes = create_dataloaders()

# Create model
print(f"\nCreating ViT model for {num_classes} classes...")
model_vanilla = create_model(num_classes)
model_vanilla = model_vanilla.to(device)

# Configure PaCA with Gradient Chain selection
peft_config = PacaConfig(
    r=48,
    target_modules=[
        "out_proj",  # Attention output projection
        "mlp.0",     # MLP first layer
        "mlp.3",     # MLP second layer
    ], 
    paca_alpha=96,
    bias="all",
    modules_to_save=["head"],
    task_type=TaskType.FEATURE_EXTRACTION,
    steps_per_epoch=len(trainloader),
    
    # Gradient Chain Configuration
    arm_distribution="gradient_chain",  # Use gradient-based chains
    num_arms=3,                         # Number of chains to create
    num_arms_to_select=1,               # Number of chains to activate per epoch
    
    # Chain construction parameters
    gradient_chain_warmup_epochs=2,     # Build chains after 2 epochs of warmup
    gradient_chain_num_samples=5,       # Average gradients over 5 batches
    
    # Arm selection method (can use UCB or Thompson Sampling)
    arm_selection_method="thompson",     # or "ucb"
    thompson_distribution="gaussian",    # Gaussian for continuous rewards
    thompson_gaussian_prior_mean=0.0,
    thompson_gaussian_prior_std=10.0,
    thompson_gaussian_noise_std=5.0,
)

print(f"\nPEFT config: {peft_config}")

model = PacaModel(model_vanilla, peft_config, adapter_name="default")

model.print_trainable_parameters()

# Model, Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine schedule with warmup
total_epochs = 25  # Fixed 25 epochs
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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
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
        
        # Calculate reward for arm selection
        if prev_val_acc is not None:
            reward = val_acc - prev_val_acc
            # Normalize reward
            normalized_reward = reward / 10.0
            model.update_ucb_rewards("default", normalized_reward)
            print(f"\nReward: {reward:.2f}% (normalized: {normalized_reward:.2f})")
        
        prev_val_acc = val_acc
        
        # Reselect arms (pass train_loader for gradient chain building)
        model.reselect_paca_weights("default", train_loader=train_loader)
        
        scheduler.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time

        print(f"\nEpoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Print gradient chain logs
        print("\nGradient Chain Statistics:")
        model.print_logs("default")
        print("-" * 60)

    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Average time per epoch: {total_time / epochs:.2f} seconds")
    
    return model


print("\n" + "=" * 60)
print("Fine-tuning ViT using Gradient Chain Selection (25 Epochs):")
print("=" * 60)

# Training
model = model.to(device)
model = train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=total_epochs)

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)

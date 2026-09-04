#!/usr/bin/env python3
"""
Vision Transformer (ViT) Fine-tuning Script for Caltech-101 with Gaussian Thompson Sampling
Rank: 8
"""
import os, time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
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
# Use absolute paths based on script location to ensure robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "{dataset_name}")
SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models_pretrained", "{model_name}")
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
    import requests
    import zipfile
    import tarfile

    # Paths
    zip_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
    data_root = os.path.join(PROJECT_ROOT, "data")
    zip_path = os.path.join(data_root, "caltech-101.zip")
    extract_root = os.path.join(data_root, "caltech-101")
    tar_path = os.path.join(extract_root, "caltech-101", "101_ObjectCategories.tar.gz")
    extract_images_dir = os.path.join(extract_root, "101_ObjectCategories")

    # Step 1: Download ZIP if not exists
    if not os.path.exists(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        print(f"Downloading Caltech-101 ZIP from {zip_url} ...")
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded to {zip_path}")

    # Step 2: Extract ZIP if not already extracted
    if not os.path.exists(tar_path):
        print(f"Extracting ZIP to {extract_root} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_root)
        print("ZIP extraction complete.")

    # Step 3: Extract .tar.gz containing images
    if not os.path.exists(extract_images_dir):
        print(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_root)
        print(f"Images extracted to {extract_images_dir}")
    else:
        print(f"Image directory already exists at {extract_images_dir}")

    # Step 4: Setup transforms
    train_transform, val_transform = get_transforms()

    # Step 5: Load ImageFolder dataset
    dataset = datasets.ImageFolder(
        root=extract_images_dir,
        transform=None  # we'll apply manually in collate
    )

    # Step 6: Train-val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    num_classes = len(dataset.classes)
    print(f"Dataset loaded: {len(train_data)} train samples, {len(val_data)} val samples")
    print(f"Number of classes: {num_classes}")

    # Step 7: Custom collate to apply transforms
    def custom_collate(transform):
        def collate_fn(batch):
            images, labels = zip(*batch)
            images = [transform(img) for img in images]
            return torch.stack(images), torch.tensor(labels)
        return collate_fn

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate(train_transform)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate(val_transform)
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
print("Vision Transformer Fine-tuning on Caltech-101")
print("Using Gaussian Thompson Sampling - Rank 8")
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

# Configure PaCA with Gaussian Thompson Sampling
peft_config = PacaConfig(
    r=8,
    target_modules=[
        "out_proj",  # Attention output projection
        "mlp.0",     # MLP first layer
        "mlp.3",     # MLP second layer
    ], 
    paca_alpha=16,
    bias="all",
    modules_to_save=["head"],
    task_type=TaskType.FEATURE_EXTRACTION,
    steps_per_epoch=len(trainloader),
    arm_distribution="horizontal",  # or "vertical" for layer-level selection
    num_arms=3,
    num_arms_to_select=1,
    
    # Gaussian Thompson Sampling configuration
    arm_selection_method="thompson",
    thompson_distribution="gaussian",  # Use Gaussian instead of Beta
    thompson_gaussian_prior_mean=0.0,  # Prior mean
    thompson_gaussian_prior_std=10.0,  # Prior std (higher = more uncertainty/exploration)
    thompson_gaussian_noise_std=5.0,   # Observation noise std
)

print(f"\nPEFT config: {peft_config}")

model = PacaModel(model_vanilla, peft_config, adapter_name="default")

model.print_trainable_parameters()

# Model, Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Cosine schedule with warmup
total_epochs = 100
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
        
        # Calculate reward for Gaussian Thompson Sampling
        if prev_val_acc is not None:
            reward = val_acc - prev_val_acc
            # Normalize reward for Gaussian (optional but recommended)
            normalized_reward = reward / 10.0  # Scale down to [-10, 10] range
            model.update_ucb_rewards("default", normalized_reward)
            print(f"\nGaussian Thompson Sampling Reward: {reward:.2f}% (normalized: {normalized_reward:.2f})")
        
        prev_val_acc = val_acc
        
        # Reselect arms using Gaussian Thompson Sampling
        model.reselect_paca_weights("default")
        
        scheduler.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time

        print(f"\nEpoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Print Gaussian Thompson Sampling logs
        print("\nGaussian Thompson Sampling Statistics:")
        model.print_logs("default")
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
                print(f"Total training time: {total_time:.2f} seconds")
                print(f"Average time per epoch: {total_time / (epoch + 1):.2f} seconds")
                break

    return model


print("\n" + "=" * 60)
print("Fine-tuning ViT using Gaussian Thompson Sampling-based PaCA:")
print("=" * 60)

# First training run with Gaussian Thompson Sampling
model = model.to(device)
model = train(model, trainloader, testloader, criterion, optimizer, scheduler, epochs=total_epochs)

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)

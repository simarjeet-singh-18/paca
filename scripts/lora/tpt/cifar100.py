#!/usr/bin/env python3
"""
Vision Transformer (ViT) Throughput Measurement Script for CIFAR-100

This script measures and compares the training throughput (images/sec) of
Full Fine-Tuning, LoRA, and PaCA on the CIFAR-100 dataset. It iterates
through various batch sizes to find the maximum throughput and the point of
Out-Of-Memory (OOM) error for each method.
"""
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel

# --- Configuration ---
# Define the batch sizes to test. The script will stop for a given method if an OOM error occurs.
BATCH_SIZES_TO_TEST = [16, 32, 64] 

# Define the methods to compare. 
# 'Full_FT' for full fine-tuning, 'LoRA' and 'PaCA' for PEFT methods.
METHODS_TO_TEST = ['LoRA']

IMG_SIZE = 224
NUM_WORKERS = 4 # Adjust based on your system's capabilities
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading Functions ---

def get_transforms():
    """Define image transformations."""
    # Simple transforms for throughput testing.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(batch_size):
    """Create a training dataloader with a specific batch size."""
    transform = get_transforms()
    
    # Download and load CIFAR100 dataset
    train_dataset = datasets.CIFAR100(
        root="./data", train=True, transform=transform, download=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return train_loader, len(train_dataset)

# --- Model Creation Function ---

def create_model(num_classes=100, method='Full_FT', trainloader=None):
    """Create and configure the Vision Transformer model based on the chosen method."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    if method == 'Full_FT':
        # Unfreeze all parameters for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    # Replace the classifier head for our number of classes
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # The head should always be trainable
    for param in model.heads.parameters():
        param.requires_grad = True

    if method == 'LoRA':
        config = LoraConfig(
            r=16,  # Rank of the low-rank adaptation matrices
            lora_alpha=32,  # Scaling factor for LoRA
            target_modules=[
                "out_proj",  # Attention output projections (12 layers)
                "mlp.0",     # MLP expansion layers (768->3072, 12 layers)
                "mlp.3",      # MLP contraction layers (3072->768, 12 layers)
            ],
            modules_to_save=["head"]  # Save the classifier head
        )
        model = get_peft_model(model, config)
    elif method == 'PaCA':
        config = PacaConfig(
            r=16,
            target_modules=[
                "out_proj",  # Attention output projections (12 layers)
                "mlp.0",     # MLP expansion layers (768->3072, 12 layers)
                "mlp.3",      # MLP contraction layers (3072->768, 12 layers)
            ],
            paca_alpha=32,
            bias="all",
            modules_to_save=["head"],
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = PacaModel(model, config, adapter_name="default")

    return model.to(DEVICE)

# --- Throughput Measurement Function ---

def measure_throughput(model, train_loader, num_samples):
    """Run one epoch of training and measure throughput."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # # Warm-up GPU
    # for _ in range(2):
    #     try:
    #         inputs, targets = next(iter(train_loader))
    #         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #     except StopIteration:
    #         print("DataLoader exhausted during warm-up. This is fine.")
    #         break

    torch.cuda.synchronize() # Wait for all GPU operations to finish
    
    start_time = time.time()
    
    # Run one full epoch
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize() # Wait for the epoch to complete
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = num_samples / total_time
    
    return throughput

# --- Plotting Function ---

def plot_results(results):
    """Generate and save a plot of the throughput results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    for method, data in results.items():
        if data: # Check if there is data to plot
            batch_sizes = [item['batch_size'] for item in data]
            throughputs = [item['throughput'] for item in data]
            ax.plot(batch_sizes, throughputs, marker='o', linestyle='-', label=method)
            
            # Mark OOM point if it occurred
            if len(batch_sizes) < len(BATCH_SIZES_TO_TEST):
                 ax.text(batch_sizes[-1] * 1.05, throughputs[-1], 'OOM', color='red', fontsize=12, va='center')


    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Throughput (images/sec)", fontsize=14)
    ax.set_title("ViT Fine-Tuning Throughput on CIFAR-100 (A100 GPU)", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_xticks(BATCH_SIZES_TO_TEST)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    save_path = "throughput_paca.png"
    plt.savefig(save_path)
    print(f"\n📈 Plot saved to {save_path}")

# --- Main Execution Logic ---

def main():
    """Main function to run the throughput analysis."""
    print("=" * 60)
    print("Starting Throughput Analysis for ViT on CIFAR-100")
    print("=" * 60)
    
    results = {method: [] for method in METHODS_TO_TEST}

    for method in METHODS_TO_TEST:
        print(f"\n--- Testing Method: {method} ---")
        for batch_size in BATCH_SIZES_TO_TEST:
            print(f"  > Trying Batch Size: {batch_size}", end="")
            
            # Clear CUDA cache to prevent carry-over memory issues
            torch.cuda.empty_cache()
            
            try:
                # 1. Create Dataloader
                train_loader, num_samples = create_dataloader(batch_size)
                
                # 2. Create Model
                model = create_model(num_classes=100, method=method, trainloader=train_loader)
                
                # 3. Measure Throughput
                throughput = measure_throughput(model, train_loader, num_samples)
                
                print(f"  |  Throughput: {throughput:.2f} images/sec")
                results[method].append({'batch_size': batch_size, 'throughput': throughput})
                
                # Clean up to release memory
                del model
                del train_loader
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("  |  Out of Memory (OOM). Stopping here for this method.")
                    break # Stop increasing batch size for this method
                else:
                    print(f"  |  An unexpected runtime error occurred: {e}")
                    break
            except Exception as e:
                print(f"  |  An unexpected error occurred: {e}")
                break

    print("\n--- Analysis Complete ---")
    for method, data in results.items():
        print(f"Results for {method}: {data}")

    plot_results(results)

if __name__ == "__main__":
    main()

# Gradient Chain Selection for PaCA PEFT

## 🎯 Overview

**Gradient Chain Selection** is a novel arm selection strategy that creates "neural pathways" through the network based on activation gradients. Unlike horizontal (feature-level) or vertical (layer-level) distributions, gradient chains follow the flow of information through the network, selecting weights that form high-importance connections across layers.

## 🧠 Core Concept

### Traditional Approaches
- **Horizontal**: Divides features within each layer into groups
- **Vertical**: Groups entire layers together

### Gradient Chain Approach (New!)
- **Chains**: Creates paths of connected neurons through layers
- **Gradient-Based**: Follows ∂(activation)/∂(weight) to find important connections
- **Fixed Structure**: Builds chains once after warmup, then keeps them fixed
- **Multi-Armed Bandits**: Uses UCB or Thompson Sampling to select which chains to train

## 📐 Algorithm

### Phase 1: Warmup (Epochs 1-3)
Train normally with random weight selection to establish initial activations.

### Phase 2: Chain Construction (After Warmup)

For each arm (chain) to create:

1. **Start at Layer 2** (first hidden layer)
2. **Select starting neuron** `i` in layer 2
3. **For each subsequent layer**:
   - Compute `∂(activation_j)/∂(w_{i,j})` for all weights `w_{i,j}` connecting neuron `i` to neurons in next layer
   - Select weight with **maximum gradient magnitude**
   - The selected weight determines which neuron `j` is in the chain
   - Move to neuron `j` in the next layer
4. **Repeat** until reaching the final layer

This creates a "string" of weights forming a path: Layer2 → Layer3 → Layer4 → ... → Final Layer

### Phase 3: Training with Chains

Use UCB or Thompson Sampling to select which chains to activate each epoch:

```python
# Each epoch:
1. Sample from arms (chains) using UCB/Thompson
2. Activate selected chains
3. Train only the weights in active chains
4. Observe reward (validation accuracy change)
5. Update arm statistics
6. Repeat
```

## 🔬 Mathematical Foundation

### Gradient Computation

For weight `w_{i,j}` connecting neuron `i` (layer L) to neuron `j` (layer L+1):

```
∂(activation_j)/∂(w_{i,j}) ≈ activation_i × gradient_j
```

Where:
- `activation_i` = output of neuron `i` in layer L
- `gradient_j` = gradient flowing back to neuron `j` in layer L+1

### Chain Selection Criterion

```python
For neuron i in layer L:
  selected_j = argmax_j |∂(activation_j)/∂(w_{i,j})|
```

This selects the **most influential** connection from neuron `i`.

### Aggregation Across Batches

To reduce noise, average gradients over multiple batches:

```python
avg_gradient = mean([gradient_batch1, gradient_batch2, ..., gradient_batchN])
```

## ⚙️ Configuration

### Basic Setup

```python
from peft import PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel

peft_config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    
    # Gradient Chain Configuration
    arm_distribution="gradient_chain",  # KEY: Use gradient chains
    num_arms=8,                         # Number of chains to create
    num_arms_to_select=2,               # Chains to activate per epoch
    
    # Chain Construction Parameters
    gradient_chain_warmup_epochs=3,     # Epochs before building chains
    gradient_chain_num_samples=5,       # Batches to average for gradients
    
    # Arm Selection Method
    arm_selection_method="thompson",     # or "ucb"
    thompson_distribution="gaussian",
    thompson_gaussian_prior_mean=0.0,
    thompson_gaussian_prior_std=10.0,
    thompson_gaussian_noise_std=5.0,
)

model = PacaModel(base_model, peft_config, adapter_name="default")
```

### Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arm_distribution` | `"horizontal"` | Set to `"gradient_chain"` for this method |
| `num_arms` | `8` | Number of chains to create |
| `num_arms_to_select` | `2` | How many chains to activate per epoch |
| `gradient_chain_warmup_epochs` | `3` | Epochs to train before building chains |
| `gradient_chain_num_samples` | `5` | Batches to average for gradient estimation |

### Arm Selection Methods

#### Option 1: Thompson Sampling with Gaussian (Recommended)

```python
arm_selection_method="thompson"
thompson_distribution="gaussian"
thompson_gaussian_prior_mean=0.0    # Neutral prior
thompson_gaussian_prior_std=10.0     # High exploration
thompson_gaussian_noise_std=5.0      # Moderate noise
```

**Why Gaussian?** Continuous rewards (accuracy changes) fit Gaussian distributions better than Beta.

#### Option 2: Thompson Sampling with Beta

```python
arm_selection_method="thompson"
thompson_distribution="beta"
thompson_alpha_prior=1.0   # Uniform prior
thompson_beta_prior=1.0
```

#### Option 3: UCB

```python
arm_selection_method="ucb"
ucb_alpha=0.5              # Exploration parameter
```

## 📊 Training Loop Integration

### Critical Change: Pass `train_loader` to `reselect_paca_weights`

```python
def train(model, train_loader, val_loader, epochs):
    prev_val_acc = None
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        for inputs, targets in train_loader:
            # ... standard training ...
            pass
        
        # Validation
        val_acc = evaluate(model, val_loader)
        
        # Calculate reward
        if prev_val_acc is not None:
            reward = val_acc - prev_val_acc
            normalized_reward = reward / 10.0  # Normalize
            model.update_ucb_rewards("default", normalized_reward)
        
        prev_val_acc = val_acc
        
        # IMPORTANT: Pass train_loader for chain building
        model.reselect_paca_weights("default", train_loader=train_loader)
        
        # Print chain statistics
        model.print_logs("default")
```

### Why Pass `train_loader`?

During warmup epochs, chains haven't been built yet. When `current_epoch >= gradient_chain_warmup_epochs`, the manager needs access to the training data to:
1. Run forward passes to get activations
2. Compute gradients
3. Average over multiple batches
4. Build the chains

## 📈 Output Logs

### During Warmup

```
Gradient chain warmup: 1/3
Gradient chain warmup: 2/3
Gradient chain warmup: 3/3
```

### Chain Construction

```
============================================================
Building Gradient Chains...
============================================================
Sampling 5 batches to estimate gradients...
  Batch 1/5 processed
  Batch 2/5 processed
  ...

Averaging gradients across batches...

Building 8 gradient chains...

Chain 0: Starting from neuron 15
  Layer 0 (encoder.layers.encoder_layer_0.mlp.0): input_col=15 -> output_neuron=127 (grad=0.0234)
  Layer 1 (encoder.layers.encoder_layer_0.mlp.3): input_col=127 -> output_neuron=42 (grad=0.0189)
  Layer 2 (encoder.layers.encoder_layer_1.mlp.0): input_col=42 -> output_neuron=201 (grad=0.0156)
  ...

Chain 1: Starting from neuron 7
  ...

✓ Successfully built 8 gradient chains
============================================================
```

### After Chain Construction

```
[PacaGradientChainManager - Gradient Chain Logs]
  Chains Built: True
  Number of Chains: 8
  Current Epoch: 5
  Selected Arms: [1, 3]
  Arm Means: [2.34, 5.12, 1.89, 4.23, 0.67, 3.45, 2.11, 1.98]
  Arm Stds: [3.21, 2.45, 4.12, 2.78, 5.34, 3.01, 3.89, 4.56]
  Arm Observation Counts: [2, 3, 1, 2, 1, 2, 1, 1]
  Last Sampled Values: [3.45, 6.12, 0.87, 5.34, -1.23, 4.56, 1.23, 3.67]
```

## 🎯 Advantages

### 1. **Biological Inspiration**
Mimics how neural networks learn dominant pathways through repeated activation.

### 2. **Information Flow**
Follows the actual flow of information through the network, not arbitrary divisions.

### 3. **Reduced Interference**
Each chain is a sparse path through the network, minimizing parameter interference.

### 4. **Gradient-Based**
Uses actual gradient information to identify important connections.

### 5. **Fixed Structure**
Once built, chains are fixed, ensuring consistent comparisons during bandit selection.

## 📉 Comparison with Other Methods

| Method | Granularity | Structure | Adaptation |
|--------|-------------|-----------|------------|
| **Horizontal** | Within layers | Feature groups | Dynamic each epoch |
| **Vertical** | Across layers | Layer groups | Dynamic each epoch |
| **Gradient Chain** | Cross-layer paths | Neural pathways | Fixed after warmup |

### When to Use Each

- **Horizontal**: When different feature subspaces are important
- **Vertical**: When different depth levels matter
- **Gradient Chain**: When information flow patterns are key

## 🔧 Hyperparameter Tuning

### Warmup Epochs

```python
gradient_chain_warmup_epochs=3  # Default

# Too low (1-2): Activations not stabilized, noisy chains
# Sweet spot (3-5): Good balance
# Too high (>10): Wasted training time
```

**Recommendation**: 3-5 epochs for most tasks.

### Number of Samples

```python
gradient_chain_num_samples=5  # Default

# Too low (1-2): Noisy gradient estimates
# Sweet spot (5-10): Good averaging
# Too high (>20): Diminishing returns, slow chain building
```

**Recommendation**: 5 batches for stability.

### Number of Arms (Chains)

```python
num_arms=8  # Default

# Fewer chains (2-4): Less exploration, faster convergence
# More chains (8-16): Better coverage, more exploration
```

**Recommendation**: 
- Small models: 4-8 chains
- Large models: 8-16 chains

### Chains to Select

```python
num_arms_to_select=2  # Default

# Fewer (1): Pure exploitation
# More (3-4): More parameter coverage
```

**Recommendation**: `num_arms_to_select = num_arms / 4`

## 🚀 Example: Vision Transformer on CIFAR-100

```python
import torch
from torchvision import models, datasets, transforms
from peft import PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel

# Load model
model = models.vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, 100)

# Configure gradient chains
config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    arm_distribution="gradient_chain",
    num_arms=8,
    num_arms_to_select=2,
    gradient_chain_warmup_epochs=3,
    gradient_chain_num_samples=5,
    arm_selection_method="thompson",
    thompson_distribution="gaussian",
    thompson_gaussian_prior_mean=0.0,
    thompson_gaussian_prior_std=10.0,
    thompson_gaussian_noise_std=5.0,
)

# Create PaCA model
peft_model = PacaModel(model, config, "default")

# Training loop
for epoch in range(100):
    # Train...
    val_acc = validate()
    reward = val_acc - prev_acc
    peft_model.update_ucb_rewards("default", reward / 10.0)
    peft_model.reselect_paca_weights("default", train_loader=train_loader)  # KEY!
```

## 🐛 Troubleshooting

### Issue: "Warning: train_loader required for gradient chain building"

**Cause**: Called `reselect_paca_weights()` without `train_loader` during chain building.

**Fix**:
```python
# Wrong
model.reselect_paca_weights("default")

# Correct
model.reselect_paca_weights("default", train_loader=train_loader)
```

### Issue: Chains not being built

**Cause**: `current_epoch < gradient_chain_warmup_epochs`

**Fix**: Wait for warmup to complete. Check logs for "Gradient chain warmup: X/Y".

### Issue: Out of memory during chain building

**Cause**: Too many samples or large batch size.

**Fix**: Reduce `gradient_chain_num_samples` or use smaller batches temporarily.

## 📚 Research Background

### Inspiration

This method is inspired by:
1. **Hebbian Learning**: "Neurons that fire together, wire together"
2. **Neural Pathway Pruning**: Biological networks strengthen important pathways
3. **Gradient Flow Analysis**: Understanding which connections matter most

### Key Insight

Instead of arbitrarily dividing the network, follow the **actual information flow** as revealed by gradients during training.

## 🎓 Best Practices

1. **Always normalize rewards** for Thompson Sampling
2. **Use Gaussian distribution** for accuracy-based tasks
3. **Monitor chain diversity** - if all chains start from same neurons, increase `num_arms`
4. **Visualize chains** to understand network structure
5. **Compare with horizontal** mode to validate benefits

## 🔮 Future Enhancements

Potential improvements:
- **Dynamic chain recomputation**: Rebuild chains periodically
- **Multi-scale chains**: Different chain lengths for different abstraction levels
- **Ensemble chains**: Combine multiple chains for prediction
- **Attention-weighted chains**: Use attention scores for chain construction

---

## 📝 Summary

**Gradient Chain Selection** creates neural pathways through the network by:
- 🔥 Following activation gradients
- 🎯 Starting from hidden layers (Layer 2+)
- 🔗 Selecting one weight per layer
- 🎲 Using bandits (UCB/Thompson) to choose chains
- 📊 Training sparse pathways end-to-end

This approach provides a **biologically-inspired**, **gradient-aware** alternative to traditional weight selection strategies!

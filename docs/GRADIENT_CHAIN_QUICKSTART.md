# Quick Start: Gradient Chain Selection

## ⚡ 30-Second Setup

```python
from peft import PacaConfig
from peft.tuners.paca.model import PacaModel

config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    
    # Gradient Chain Mode
    arm_distribution="gradient_chain",
    num_arms=8,
    num_arms_to_select=2,
    gradient_chain_warmup_epochs=3,
    gradient_chain_num_samples=5,
    
    # Thompson Sampling
    arm_selection_method="thompson",
    thompson_distribution="gaussian",
    thompson_gaussian_prior_std=10.0,
    thompson_gaussian_noise_std=5.0,
)

model = PacaModel(base_model, config, "default")
```

## 🔄 Training Loop

```python
for epoch in range(100):
    # Train
    train_one_epoch(model, train_loader)
    
    # Validate
    val_acc = evaluate(model, val_loader)
    
    # Update rewards
    reward = (val_acc - prev_acc) / 10.0  # Normalize!
    model.update_ucb_rewards("default", reward)
    
    # Reselect chains (MUST pass train_loader!)
    model.reselect_paca_weights("default", train_loader=train_loader)
    
    # Monitor
    model.print_logs("default")
    
    prev_acc = val_acc
```

## 📊 Key Differences from Horizontal/Vertical

| Aspect | Horizontal/Vertical | Gradient Chain |
|--------|---------------------|----------------|
| **Chain Building** | None | Required after warmup |
| **reselect_paca_weights()** | No args needed | **MUST pass `train_loader`** |
| **Structure** | Dynamic | Fixed after construction |
| **Granularity** | Layer-wise | Cross-layer paths |

## ⚠️ Common Mistakes

### ❌ Wrong
```python
model.reselect_paca_weights("default")  # Missing train_loader!
```

### ✅ Correct
```python
model.reselect_paca_weights("default", train_loader=train_loader)
```

## 🎛️ Quick Tuning Guide

```python
# More exploration → Increase prior_std
thompson_gaussian_prior_std=20.0  # Very exploratory

# Noisier rewards → Increase noise_std
thompson_gaussian_noise_std=10.0  # High noise tolerance

# Longer warmup → More stable chains
gradient_chain_warmup_epochs=5  # Wait longer

# More samples → Better gradient estimates
gradient_chain_num_samples=10  # More accurate chains
```

## 📈 Expected Output

### Epoch 1-3 (Warmup)
```
Gradient chain warmup: 1/3
Gradient chain warmup: 2/3
Gradient chain warmup: 3/3
```

### Epoch 4 (Chain Building)
```
============================================================
Building Gradient Chains...
...
✓ Successfully built 8 gradient chains
============================================================
```

### Epoch 5+ (Training with Chains)
```
[PacaGradientChainManager - Gradient Chain Logs]
  Chains Built: True
  Number of Chains: 8
  Selected Arms: [1, 3]
  Arm Means: [2.34, 5.12, ...]
  ...
```

## 🚀 Run Example

```bash
# Install modified PEFT
cd paca_modded/peft && pip install -e .

# Run example script
python scripts/gradient_chain/cifar100_gradient_chain.py
```

## 💡 When to Use

**Use Gradient Chains when:**
- ✅ You want biologically-inspired paths
- ✅ Information flow patterns matter
- ✅ You can afford warmup epochs
- ✅ You want sparse, interpretable connections

**Use Horizontal when:**
- ✅ Feature subspaces are important
- ✅ Need dynamic adaptation every epoch
- ✅ No warmup time available

**Use Vertical when:**
- ✅ Layer depth matters more than width
- ✅ Want simple layer grouping
- ✅ Fastest to set up

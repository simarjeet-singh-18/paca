# PaCA Arm Distribution Strategies Comparison

## Overview

PaCA PEFT now supports **three distinct arm distribution strategies** for weight selection:

1. **Horizontal** - Feature-level division within layers
2. **Vertical** - Layer-level grouping across depth
3. **Gradient Chain** - Neural pathway construction (NEW!)

---

## Visual Comparison

### 1. Horizontal Distribution

```
Layer 1: [●●●●●|●●●●●|●●●●●]  ← 3 arms (feature groups)
             Arm 0  Arm 1  Arm 2

Layer 2: [●●●●●|●●●●●|●●●●●]  ← 3 arms (feature groups)
             Arm 0  Arm 1  Arm 2

Layer 3: [●●●●●|●●●●●|●●●●●]  ← 3 arms (feature groups)
             Arm 0  Arm 1  Arm 2
```

**Characteristics:**
- Divides input features in each layer into groups
- Each arm = subset of features in every layer
- Dynamic: Arms reselected every epoch
- Good for: Feature importance varies

---

### 2. Vertical Distribution

```
Layer 1: [●●●●●●●●●●●●●●●]  ← Arm 0
         ─────────────────

Layer 2: [●●●●●●●●●●●●●●●]  ← Arm 1
         ─────────────────

Layer 3: [●●●●●●●●●●●●●●●]  ← Arm 2
         ─────────────────
```

**Characteristics:**
- Groups entire layers together
- Each arm = one or more complete layers
- Dynamic: Layers reselected every epoch
- Good for: Depth matters more than width

---

### 3. Gradient Chain Distribution (NEW!)

```
Start Layer 2

Chain 0:  ●
          │\
          │ \
Chain 1: ●   ●
          \  │\
           \ │ \
            ●│  ●
             \  │
              \ │
               ●●

Each chain = path through network
```

**Detailed View:**

```
Layer 2:  [ ● · · · ● · · · ]  ← Starting neurons
            │       │
            │       └─────┐
            │             │
            ↓             ↓
Layer 3:  [ · · ● · · · · ● ]  ← Selected by max gradient
            │             │
            └─┐           │
              │           │
              ↓           ↓
Layer 4:  [ · ● · · · ● · · ]  ← Selected by max gradient
              │       │
              └───────┘
                  │
                  ↓
Layer 5:  [ · · · ● · · · · ]  ← Final neuron

Chain 0: Layer2[0] → Layer3[2] → Layer4[1] → Layer5[3]
Chain 1: Layer2[4] → Layer3[7] → Layer4[5] → Layer5[3]
```

**Characteristics:**
- Creates sparse pathways through entire network
- Each arm = one weight per layer forming a chain
- Fixed: Chains built once after warmup, then frozen
- Good for: Information flow patterns matter

---

## Algorithm Comparison

### Horizontal

```python
# Each epoch
for layer in layers:
    divide features into num_arms groups
    select num_arms_to_select groups using bandit
    train only selected features
```

### Vertical

```python
# Each epoch
divide layers into num_arms groups
select num_arms_to_select layer groups using bandit
train only selected layer groups
```

### Gradient Chain

```python
# Once after warmup
for each arm:
    start at random neuron in layer 2
    for each layer:
        compute gradients for all weights
        select weight with max gradient
        add to chain
    store chain

# Each epoch thereafter
select num_arms_to_select chains using bandit
train only weights in selected chains
```

---

## Configuration Examples

### Horizontal Setup
```python
PacaConfig(
    arm_distribution="horizontal",
    num_arms=8,
    num_arms_to_select=2,
    arm_selection_method="thompson",
    thompson_distribution="gaussian",
)
```

### Vertical Setup
```python
PacaConfig(
    arm_distribution="vertical",
    num_arms=4,
    num_arms_to_select=1,
    arm_selection_method="ucb",
    ucb_alpha=0.5,
)
```

### Gradient Chain Setup
```python
PacaConfig(
    arm_distribution="gradient_chain",
    num_arms=8,
    num_arms_to_select=2,
    gradient_chain_warmup_epochs=3,
    gradient_chain_num_samples=5,
    arm_selection_method="thompson",
    thompson_distribution="gaussian",
)
```

---

## Training Loop Differences

### Horizontal / Vertical
```python
for epoch in range(total_epochs):
    train_one_epoch()
    val_acc = evaluate()
    reward = val_acc - prev_acc
    model.update_ucb_rewards("default", reward)
    model.reselect_paca_weights("default")  # No extra args
```

### Gradient Chain
```python
for epoch in range(total_epochs):
    train_one_epoch()
    val_acc = evaluate()
    reward = val_acc - prev_acc
    model.update_ucb_rewards("default", reward)
    model.reselect_paca_weights("default", train_loader=train_loader)  # ⚠️ Must pass train_loader!
```

---

## Decision Guide

### Choose **Horizontal** when:
- ✅ Different feature dimensions have varying importance
- ✅ You want fine-grained control within each layer
- ✅ Model has wide layers (many features)
- ✅ No warmup time constraints
- ✅ Dynamic adaptation every epoch is beneficial

### Choose **Vertical** when:
- ✅ Network depth is the critical factor
- ✅ Different layers serve different roles (early/mid/late features)
- ✅ You want to control computational budget by layer
- ✅ Simplest setup desired
- ✅ Model is deep (many layers)

### Choose **Gradient Chain** when:
- ✅ Information flow patterns are important
- ✅ You want biologically-inspired pathways
- ✅ Interpretability is a priority
- ✅ Can afford warmup period (3-5 epochs)
- ✅ Want sparse, cross-layer connections
- ✅ Model has both depth and width

---

## Performance Characteristics

| Strategy | Setup Complexity | Runtime Overhead | Memory Usage | Interpretability |
|----------|------------------|------------------|--------------|------------------|
| **Horizontal** | Low | Low | Low | Medium |
| **Vertical** | Low | Lowest | Lowest | High |
| **Gradient Chain** | Medium | Medium (warmup) | Medium | Very High |

---

## Combination Strategies

### Not Currently Supported (Future Work)
- Horizontal + Vertical hybrid
- Multiple gradient chains with different starting layers
- Ensemble of all three strategies

---

## Example Use Cases

### Horizontal
- **Image Classification**: Different visual features (edges, textures, shapes) matter
- **NLP**: Different embedding dimensions capture different semantic aspects
- **Time Series**: Different frequency components

### Vertical
- **Transfer Learning**: Fine-tune only later layers
- **Progressive Training**: Train shallow → deep
- **Computational Constraints**: Limit active layers to save compute

### Gradient Chain
- **Neural Architecture Search**: Identify important pathways
- **Model Compression**: Find minimal sufficient subnetworks
- **Interpretability**: Visualize information flow
- **Neuroscience**: Study learning dynamics

---

## Implementation Files

| Strategy | Files Modified | Lines of Code |
|----------|----------------|---------------|
| Horizontal | layer.py | ~300 |
| Vertical | model.py | ~200 |
| Gradient Chain | config.py, model.py | ~800 |

---

## Research Potential

### Open Questions
1. **Hybrid Approaches**: Can we combine multiple strategies?
2. **Dynamic Chains**: Should chains be rebuilt periodically?
3. **Multi-Scale**: Different chain lengths for different tasks?
4. **Attention Integration**: Use attention weights for chain construction?
5. **Transfer**: Do chains transfer across similar tasks?

---

## Quick Command Reference

```bash
# Horizontal example
python scripts/thompson/cifar100_gaussian.py

# Gradient Chain example
python scripts/gradient_chain/cifar100_gradient_chain.py

# Vertical example
# (Create similar script with arm_distribution="vertical")
```

---

## Summary Table

| Aspect | Horizontal | Vertical | Gradient Chain |
|--------|-----------|----------|----------------|
| **Granularity** | Feature-level | Layer-level | Path-level |
| **Structure** | Groups of columns | Groups of layers | Chains of neurons |
| **Selection** | Dynamic | Dynamic | Fixed after warmup |
| **Gradient-Based** | ❌ | ❌ | ✅ |
| **Biological** | ❌ | ❌ | ✅ |
| **Cross-Layer** | ❌ | ✅ | ✅ |
| **Warmup** | No | No | Yes (3-5 epochs) |
| **train_loader** | Not needed | Not needed | Needed during warmup |
| **Best For** | Feature importance | Depth importance | Flow patterns |

---

**Choose the strategy that best matches your problem structure!** 🎯

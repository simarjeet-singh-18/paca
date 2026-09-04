# Thompson Sampling Implementation for PaCA

## Overview

This implementation adds **Thompson Sampling** as an alternative arm selection method to the existing UCB (Upper Confidence Bound) approach in the PaCA PEFT method. Thompson Sampling is a Bayesian approach that uses probability matching for exploration-exploitation trade-offs.

## Key Differences: UCB vs Thompson Sampling

| Aspect | UCB | Thompson Sampling |
|--------|-----|-------------------|
| **Approach** | Deterministic confidence bounds | Probabilistic sampling from Beta distribution |
| **Parameters** | Cumulative rewards + selection counts | Successes (α) + Failures (β) for Beta(α, β) |
| **Selection Logic** | Highest UCB score: μᵢ + α√(2ln(t)/nᵢ) | Highest value sampled from Beta(α, β) |
| **Reward Update** | Accumulate rewards directly | Increment α (positive) or β (negative) |
| **Exploration** | Controlled by ucb_alpha parameter | Natural from probability distribution |
| **Convergence** | Logarithmic regret guarantees | Bayesian regret bounds |

## Implementation Architecture

### 1. Configuration (`config.py`)

**New Parameters:**
- `arm_selection_method`: `"ucb"` or `"thompson"` (default: `"ucb"`)
- `thompson_alpha_prior`: Prior for Beta distribution α (default: 1.0)
- `thompson_beta_prior`: Prior for Beta distribution β (default: 1.0)

### 2. Layer-Level (`layer.py`)

**New Attributes in `PacaLayer`:**
- `arm_selection_method`: Determines which algorithm to use
- `thompson_alpha_prior` / `thompson_beta_prior`: Priors for Beta distribution
- `arm_successes`: BufferDict storing α parameters for each adapter
- `arm_failures`: BufferDict storing β parameters for each adapter

**New Methods:**
- `reselect_thompson_indices(adapter_name)`: Samples from Beta(α, β) and selects top arms
- `update_thompson_rewards(adapter_name, reward)`: Updates α/β based on reward sign
- Modified `reselect_indices()` and `reselect_ucb_indices()`: Route based on selection method

### 3. Model-Level (`model.py`)

**New Class: `PacaThompsonManager`**
Manages vertical distribution (layer-level) Thompson Sampling:
- Divides layers into arms
- Samples from Beta distribution for each arm
- Enables/disables layer groups based on sampling
- Updates α/β parameters after each epoch

**Updated `PacaModel` Methods:**
- `__init__`: Creates Thompson or UCB manager based on config
- `reselect_paca_weights()`: Routes to Thompson or UCB method
- `update_ucb_rewards()`: Handles both UCB and Thompson updates
- `print_logs()`: Displays appropriate statistics (α/β vs rewards)

## Usage Examples

### Horizontal Distribution (Feature-Level)

```python
from peft import PacaConfig, TaskType
from peft.tuners.paca.model import PacaModel

peft_config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    bias="all",
    modules_to_save=["head"],
    task_type=TaskType.FEATURE_EXTRACTION,
    steps_per_epoch=len(trainloader),
    
    # Horizontal distribution settings
    arm_distribution="horizontal",
    num_arms=3,
    num_arms_to_select=1,
    
    # Thompson Sampling configuration
    arm_selection_method="thompson",
    thompson_alpha_prior=1.0,
    thompson_beta_prior=1.0,
)

model = PacaModel(base_model, peft_config, adapter_name="default")
```

### Vertical Distribution (Layer-Level)

```python
peft_config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    
    # Vertical distribution settings
    arm_distribution="vertical",
    num_arms=4,
    num_arms_to_select=2,
    
    # Thompson Sampling configuration
    arm_selection_method="thompson",
    thompson_alpha_prior=1.0,
    thompson_beta_prior=1.0,
)

model = PacaModel(base_model, peft_config, adapter_name="default")
```

### Training Loop Integration

```python
def train_epoch(model, train_loader, val_loader, optimizer):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    val_acc = evaluate(model, val_loader)
    
    # Calculate reward (improvement in validation accuracy)
    reward = val_acc - prev_val_acc
    
    # Update Thompson Sampling parameters
    model.update_ucb_rewards("default", reward)
    
    # Reselect arms for next epoch
    model.reselect_paca_weights("default")
    
    # Print statistics
    model.print_logs("default")
    
    return val_acc
```

## Thompson Sampling Algorithm Details

### Selection Phase

For each arm i, sample from Beta distribution:
```
θᵢ ~ Beta(αᵢ, βᵢ)
```

Select top k arms with highest sampled values:
```
selected_arms = argmax_k(θ₁, θ₂, ..., θₙ)
```

### Update Phase

After observing reward r:
- If r > 0 (success): `αᵢ ← αᵢ + |r|`
- If r ≤ 0 (failure): `βᵢ ← βᵢ + |r|`

### Interpretation

- **α (successes)**: Higher α → arm believed to be better
- **β (failures)**: Higher β → arm believed to be worse
- **Sampling**: Naturally balances exploration vs exploitation
  - High uncertainty (α≈β) → More exploration
  - High confidence (α>>β or β>>α) → More exploitation

## Output Logs

### Thompson Sampling Logs
```
Thompson Sampling Statistics:
  Selected Arms: [1]
  Arm Successes (Alpha): [2.45, 3.87, 1.23]
  Arm Failures (Beta): [1.0, 1.0, 2.34]
  Last Sampled Values: [0.712, 0.854, 0.391]
  Arm Counts: [15, 23, 12]
  Value of t: 50
```

### UCB Logs (for comparison)
```
UCB Statistics:
  Selected Arms: [1]
  Arm Rewards: [12.5, 18.3, 5.2]
  UCB Scores: [0.923, 1.145, 0.687]
  Arm Counts: [15, 23, 12]
  Value of t: 50
```

## Hyperparameter Tuning Guide

### `thompson_alpha_prior` & `thompson_beta_prior`

**Uniform Prior (α=1, β=1):**
- Default setting
- No initial bias
- Pure data-driven learning

**Optimistic Prior (α>β):**
- Example: α=2, β=1
- Encourages initial exploration
- Good for unknown domains

**Pessimistic Prior (α<β):**
- Example: α=1, β=2
- Conservative exploration
- Good when errors are costly

**Informative Prior (α=β>1):**
- Example: α=5, β=5
- Reduces initial variance
- Good with prior knowledge

### `num_arms` & `num_arms_to_select`

**More Arms (num_arms ↑):**
- Finer-grained control
- More exploration needed
- Better for large models

**Fewer Arms (num_arms ↓):**
- Faster convergence
- Less overhead
- Better for small models

**Selection Ratio:**
- `num_arms_to_select / num_arms = 0.3-0.5` → Balanced
- Ratio < 0.3 → More exploitation
- Ratio > 0.5 → More exploration

## Files Modified

1. **`paca_rl/paca_modded/peft/src/peft/tuners/paca/config.py`**
   - Added `arm_selection_method`, `thompson_alpha_prior`, `thompson_beta_prior`
   - Added validation for Thompson Sampling parameters

2. **`paca_rl/paca_modded/peft/src/peft/tuners/paca/layer.py`**
   - Added Thompson Sampling attributes to `PacaLayer`
   - Implemented `reselect_thompson_indices()` and `update_thompson_rewards()`
   - Updated routing logic in `reselect_indices()` and `reselect_ucb_indices()`
   - Updated `Linear.__init__()` to accept Thompson parameters

3. **`paca_rl/paca_modded/peft/src/peft/tuners/paca/model.py`**
   - Added `PacaThompsonManager` class for vertical distribution
   - Updated `PacaModel.__init__()` to create appropriate manager
   - Modified `reselect_paca_weights()` and `update_ucb_rewards()` for routing
   - Enhanced `print_logs()` to display Thompson statistics

4. **`paca_rl/scripts/thompson/cifar100.py`**
   - Example script demonstrating Thompson Sampling usage
   - Includes complete training loop with reward calculation

## Testing

All files have been syntax-checked and compile successfully:

```bash
cd /export/home/achyut/sam/paca_rl/paca_modded/peft/src
python -m py_compile peft/tuners/paca/config.py  # ✓ Success
python -m py_compile peft/tuners/paca/layer.py   # ✓ Success
python -m py_compile peft/tuners/paca/model.py   # ✓ Success

cd /export/home/achyut/sam/paca_rl/scripts/thompson
python -m py_compile cifar100.py                 # ✓ Success
```

## Advantages of Thompson Sampling

1. **Bayesian Framework**: Naturally incorporates uncertainty
2. **Probability Matching**: Optimal exploration-exploitation balance
3. **Parameter-Free Exploration**: No need to tune exploration parameter (like ucb_alpha)
4. **Better Empirical Performance**: Often outperforms UCB in practice
5. **Interpretable**: α and β have clear probabilistic meaning

## When to Use Thompson Sampling vs UCB

**Use Thompson Sampling when:**
- You want more adaptive exploration
- Reward signals are noisy
- You prefer probabilistic decisions
- You have prior knowledge (can set α, β priors)

**Use UCB when:**
- You need deterministic behavior
- You want theoretical guarantees
- Reward signals are stable
- You need to tune exploration explicitly

## References

- Thompson, W. R. (1933). "On the likelihood that one unknown probability exceeds another"
- Agrawal, S. & Goyal, N. (2012). "Analysis of Thompson Sampling for the Multi-armed Bandit Problem"
- Russo, D. et al. (2018). "A Tutorial on Thompson Sampling"

## Future Enhancements

Potential improvements:
1. **Contextual Thompson Sampling**: Use contextual information for better decisions
2. **Adaptive Priors**: Learn α, β priors from meta-learning
3. **Non-stationary Bandits**: Discount old observations for changing environments
4. **Correlated Arms**: Model correlations between arms for better generalization

# Gaussian Thompson Sampling Implementation for PaCA

## Overview

This implementation adds **Gaussian Thompson Sampling** as an alternative to Beta Thompson Sampling for the PaCA PEFT method. Both distribution types are now available and can be selected via configuration.

## Comparison: Beta vs Gaussian Thompson Sampling

| Aspect | Beta Distribution | Gaussian Distribution |
|--------|-------------------|----------------------|
| **Best For** | Binary/discrete rewards (success/failure) | Continuous rewards (accuracy changes) |
| **Parameters** | α (successes), β (failures) | μ (mean), σ² (variance) |
| **Update Rule** | Increment α or β based on reward sign | Bayesian posterior update |
| **Prior** | Beta(α₀, β₀) - usually (1, 1) | Normal(μ₀, σ₀²) |
| **Sampling** | Sample from Beta(α, β) ∈ [0, 1] | Sample from Normal(μ, σ²) ∈ ℝ |
| **Uncertainty** | High when α≈β, low when α≫β or β≫α | High σ² = more exploration |
| **Convergence** | Can get stuck with early lucky wins | Smoother convergence with proper priors |

## When to Use Each Distribution

### Use **Beta Distribution** When:
- ✓ Rewards are binary (success/failure, win/loss)
- ✓ Rewards are percentages or probabilities (0-100%)
- ✓ You want simple, interpretable parameters
- ✓ Traditional Thompson Sampling behavior

### Use **Gaussian Distribution** When:
- ✓ Rewards are continuous (accuracy improvements, loss reductions)
- ✓ Rewards can be negative (performance degradation)
- ✓ You want better handling of noisy rewards
- ✓ You need more stable exploration with proper priors
- ✓ **Recommended for accuracy-based tasks!**

## Configuration

### Beta Thompson Sampling (Original)

```python
peft_config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    steps_per_epoch=len(trainloader),
    
    # Thompson Sampling with Beta distribution
    arm_distribution="horizontal",
    num_arms=3,
    num_arms_to_select=1,
    arm_selection_method="thompson",
    thompson_distribution="beta",  # Use Beta distribution
    thompson_alpha_prior=1.0,      # Success prior
    thompson_beta_prior=1.0,       # Failure prior
)
```

### Gaussian Thompson Sampling (New)

```python
peft_config = PacaConfig(
    r=48,
    target_modules=["out_proj", "mlp.0", "mlp.3"],
    paca_alpha=96,
    steps_per_epoch=len(trainloader),
    
    # Thompson Sampling with Gaussian distribution
    arm_distribution="horizontal",
    num_arms=3,
    num_arms_to_select=1,
    arm_selection_method="thompson",
    thompson_distribution="gaussian",    # Use Gaussian distribution
    thompson_gaussian_prior_mean=0.0,    # Prior mean (expected reward)
    thompson_gaussian_prior_std=10.0,    # Prior std (uncertainty)
    thompson_gaussian_noise_std=5.0,     # Observation noise std
)
```

## Algorithm Details

### Gaussian Thompson Sampling Mathematics

**Initialization:**
```
For each arm i:
  μᵢ ~ Normal(μ₀, σ₀²)  # Prior mean and variance
```

**Selection Phase:**
```
For each arm i:
  θᵢ ~ Normal(μᵢ, σᵢ²)  # Sample from posterior
  
Select arms with highest sampled values θᵢ
```

**Update Phase (Bayesian Posterior):**
```
Given observation reward r with noise variance σ_noise²:

Prior precision:        τ_prior = 1/σᵢ²
Observation precision:  τ_obs = 1/σ_noise²

Posterior precision:    τ_post = τ_prior + τ_obs
Posterior variance:     σ_post² = 1/τ_post

Posterior mean:         μ_post = σ_post² * (μᵢ/σᵢ² + r/σ_noise²)

Update: μᵢ ← μ_post, σᵢ² ← σ_post²
```

### Key Advantages of Gaussian

1. **Natural for Continuous Rewards**: Accuracy changes are continuous, not binary
2. **Bayesian Updating**: Elegant mathematical framework for uncertainty
3. **Handles Negative Rewards**: Can represent performance degradation naturally
4. **Confidence Intervals**: σᵢ directly represents uncertainty in arm quality
5. **Smoother Exploration**: Less prone to premature convergence than Beta

## Hyperparameter Tuning Guide

### `thompson_gaussian_prior_mean`

**Default: 0.0** (neutral prior)

- Set to **0.0** if you have no prior belief about arm quality
- Set **> 0** if you believe arms will generally improve performance (optimistic)
- Set **< 0** if you're conservative (pessimistic)

**Examples:**
```python
thompson_gaussian_prior_mean=0.0   # Neutral (recommended)
thompson_gaussian_prior_mean=5.0   # Optimistic (expect +5% improvement)
thompson_gaussian_prior_mean=-2.0  # Pessimistic (expect -2% change)
```

### `thompson_gaussian_prior_std`

**Default: 10.0** (high uncertainty)

Controls initial exploration:
- **Higher (10-20)**: More exploration early on
- **Lower (1-5)**: More exploitation, trust prior mean
- **Very high (>20)**: Essentially uniform exploration

**Examples:**
```python
thompson_gaussian_prior_std=20.0  # Aggressive exploration
thompson_gaussian_prior_std=10.0  # Balanced (recommended)
thompson_gaussian_prior_std=5.0   # Conservative, trust prior
thompson_gaussian_prior_std=1.0   # Minimal exploration
```

### `thompson_gaussian_noise_std`

**Default: 5.0** (moderate noise)

Models observation noise/variance:
- **Higher (5-10)**: Assumes rewards are noisy, slower learning
- **Lower (1-3)**: Assumes rewards are reliable, faster learning
- Reflects confidence in individual observations

**Examples:**
```python
thompson_gaussian_noise_std=10.0  # Very noisy rewards
thompson_gaussian_noise_std=5.0   # Moderate noise (recommended)
thompson_gaussian_noise_std=2.0   # Reliable rewards
thompson_gaussian_noise_std=1.0   # Very reliable rewards
```

## Recommended Configurations

### For Accuracy-Based Tasks (CIFAR-100, ImageNet, etc.)

**Balanced Exploration:**
```python
arm_selection_method="thompson"
thompson_distribution="gaussian"
thompson_gaussian_prior_mean=0.0   # Neutral
thompson_gaussian_prior_std=10.0   # Good exploration
thompson_gaussian_noise_std=5.0    # Moderate noise assumption
```

**Aggressive Exploration (many similar arms):**
```python
thompson_gaussian_prior_mean=0.0
thompson_gaussian_prior_std=20.0   # High uncertainty
thompson_gaussian_noise_std=10.0   # High noise tolerance
```

**Conservative (few arms, want stability):**
```python
thompson_gaussian_prior_mean=0.0
thompson_gaussian_prior_std=5.0    # Lower uncertainty
thompson_gaussian_noise_std=2.0    # Trust observations
```

## Training Loop Integration

```python
def train_epoch(model, train_loader, val_loader, criterion, optimizer):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    val_acc = evaluate(model, val_loader)
    
    # Calculate reward (continuous, can be negative)
    reward = val_acc - previous_val_acc
    
    # Optional: Normalize reward to reasonable range
    normalized_reward = reward / 10.0  # Scale to [-10, 10]
    
    # Update Gaussian parameters
    model.update_ucb_rewards("default", normalized_reward)
    
    # Reselect arms
    model.reselect_paca_weights("default")
    
    # Print statistics
    model.print_logs("default")
    
    return val_acc
```

## Output Logs

### Gaussian Thompson Sampling Logs

```
Gaussian Thompson Sampling Statistics:
  Selected Arms: [1]
  Arm Means: [2.34, 4.56, 1.23]         # Posterior means (expected rewards)
  Arm Stds: [3.21, 2.45, 4.12]          # Posterior stds (uncertainty)
  Arm Observation Counts: [5, 8, 3]     # Number of times observed
  Last Sampled Values: [2.89, 5.12, 0.87]  # Actual samples used for selection
  Arm Counts: [15, 23, 12]              # Selection frequency
  Value of t: 50                         # Total epochs
```

**Interpretation:**
- **Arm Means**: Higher mean = arm believed to be better
- **Arm Stds**: Higher std = more uncertain about arm quality
- **Sample Values**: Actual realizations from Normal(mean, std²)
- Arm with highest sample wins selection (not necessarily highest mean!)

### Beta Thompson Sampling Logs (for comparison)

```
Thompson Sampling Statistics:
  Selected Arms: [1]
  Arm Successes (Alpha): [2.45, 3.87, 1.23]
  Arm Failures (Beta): [1.0, 1.0, 2.34]
  Last Sampled Values: [0.712, 0.854, 0.391]
  Arm Counts: [15, 23, 12]
  Value of t: 50
```

## Implementation Files Modified

1. **`config.py`**: Added Gaussian parameters and validation
2. **`layer.py`**: 
   - Added Gaussian attributes (`arm_means`, `arm_stds`, `arm_observation_counts`)
   - Implemented `reselect_thompson_gaussian_indices()`
   - Implemented `update_thompson_gaussian_rewards()`
   - Updated routing logic to handle both distributions
3. **`model.py`**:
   - Added `PacaThompsonGaussianManager` class
   - Updated initialization to create appropriate manager
   - Updated reward/selection methods to route correctly
   - Enhanced logging to display Gaussian statistics

## Example Scripts

- **`scripts/thompson/cifar100.py`**: Beta Thompson Sampling example
- **`scripts/thompson/cifar100_gaussian.py`**: Gaussian Thompson Sampling example

## Mathematical Comparison

### Beta Distribution Updates

```python
# Positive reward → increase alpha
if reward > 0:
    alpha += abs(reward)
# Negative reward → increase beta
else:
    beta += abs(reward)
```

**Problem**: Large rewards dominate, hard to recover from early lucky wins

### Gaussian Distribution Updates

```python
# Bayesian posterior update
tau_prior = 1 / sigma²
tau_obs = 1 / noise_sigma²
tau_post = tau_prior + tau_obs

sigma_post² = 1 / tau_post
mu_post = sigma_post² * (mu/sigma² + reward/noise_sigma²)
```

**Advantage**: Weighted average considers both prior and observation with proper uncertainty

## Performance Expectations

### Exploration Behavior

**Beta Distribution:**
- Can converge too quickly with large early rewards
- Example: +70% reward → Alpha=71, Beta=1 → samples near 0.99
- Other arms rarely explored after lucky start

**Gaussian Distribution:**
- Smoother convergence due to variance shrinkage
- Example: +70% reward with noise_std=5, prior_std=10
  - Prior: Normal(0, 100)
  - Posterior: Normal(~3.3, ~4.8)
  - Still significant uncertainty, continued exploration

### Expected Results

With proper tuning, Gaussian Thompson Sampling should:
- ✓ Explore all arms more uniformly
- ✓ Adapt better to noisy rewards
- ✓ Achieve better final performance
- ✓ Be more robust to hyperparameter choices

## Tips for Best Results

1. **Start with neutral prior**: `prior_mean=0.0`
2. **Use high prior std for exploration**: `prior_std=10-20`
3. **Match noise std to reward scale**: If rewards are ±5%, use `noise_std=5`
4. **Normalize rewards**: Scale to reasonable range (e.g., [-10, 10])
5. **Monitor arm statistics**: Watch means and stds in logs
6. **Increase prior std if**: Arms not being explored enough
7. **Decrease noise std if**: Convergence too slow

## Future Enhancements

Potential improvements:
1. **Adaptive noise estimation**: Learn σ_noise² from data
2. **Non-stationary Gaussians**: Add forgetting factor for changing environments
3. **Contextual Gaussian TS**: Incorporate contextual information
4. **Multi-task priors**: Share information across related tasks

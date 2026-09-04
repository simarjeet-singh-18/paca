#!/usr/bin/env python3
"""
Comparison script to demonstrate UCB vs Thompson Sampling side-by-side
"""

# Example configuration for UCB
ucb_config = {
    "arm_selection_method": "ucb",
    "num_arms": 3,
    "num_arms_to_select": 1,
    "ucb_alpha": 6,  # Exploration parameter
}

# Example configuration for Thompson Sampling
thompson_config = {
    "arm_selection_method": "thompson",
    "num_arms": 3,
    "num_arms_to_select": 1,
    "thompson_alpha_prior": 1.0,  # Success prior
    "thompson_beta_prior": 1.0,   # Failure prior
}

# Optimistic Thompson Sampling (encourages exploration)
thompson_optimistic_config = {
    "arm_selection_method": "thompson",
    "num_arms": 3,
    "num_arms_to_select": 1,
    "thompson_alpha_prior": 2.0,  # Higher success prior
    "thompson_beta_prior": 1.0,
}

# Conservative Thompson Sampling (cautious exploration)
thompson_conservative_config = {
    "arm_selection_method": "thompson",
    "num_arms": 3,
    "num_arms_to_select": 1,
    "thompson_alpha_prior": 1.0,
    "thompson_beta_prior": 2.0,   # Higher failure prior
}

# High confidence Thompson Sampling (lower variance)
thompson_confident_config = {
    "arm_selection_method": "thompson",
    "num_arms": 3,
    "num_arms_to_select": 1,
    "thompson_alpha_prior": 5.0,  # Both high
    "thompson_beta_prior": 5.0,   # Reduces initial variance
}

print("="*70)
print("PaCA Arm Selection Methods Comparison")
print("="*70)

print("\n1. UCB (Upper Confidence Bound)")
print("-" * 70)
print("Configuration:", ucb_config)
print("\nCharacteristics:")
print("  - Deterministic selection based on confidence bounds")
print("  - Exploration controlled by ucb_alpha parameter")
print("  - Formula: μᵢ + α√(2ln(t)/nᵢ)")
print("  - Good for: Stable rewards, need for theoretical guarantees")
print("\nOutput Example:")
print("  Selected Arms: [1]")
print("  Arm Rewards: [12.5, 18.3, 5.2]")
print("  UCB Scores: [0.923, 1.145, 0.687]")
print("  Arm Counts: [15, 23, 12]")

print("\n\n2. Thompson Sampling (Uniform Prior)")
print("-" * 70)
print("Configuration:", thompson_config)
print("\nCharacteristics:")
print("  - Probabilistic selection from Beta(α, β) distribution")
print("  - Natural exploration-exploitation balance")
print("  - No need to tune exploration parameter")
print("  - Good for: Noisy rewards, adaptive exploration")
print("\nOutput Example:")
print("  Selected Arms: [1]")
print("  Arm Successes (Alpha): [2.45, 3.87, 1.23]")
print("  Arm Failures (Beta): [1.0, 1.0, 2.34]")
print("  Last Sampled Values: [0.712, 0.854, 0.391]")
print("  Arm Counts: [15, 23, 12]")

print("\n\n3. Thompson Sampling (Optimistic Prior)")
print("-" * 70)
print("Configuration:", thompson_optimistic_config)
print("\nCharacteristics:")
print("  - Higher α prior encourages initial exploration")
print("  - Assumes arms are good until proven otherwise")
print("  - Good for: Unknown domains, want to try all options")
print("\nWhen to use:")
print("  - Starting fresh with no prior knowledge")
print("  - Cost of exploration is low")
print("  - Want aggressive initial exploration")

print("\n\n4. Thompson Sampling (Conservative Prior)")
print("-" * 70)
print("Configuration:", thompson_conservative_config)
print("\nCharacteristics:")
print("  - Higher β prior reduces initial exploration")
print("  - Assumes arms are bad until proven otherwise")
print("  - Good for: When errors are costly")
print("\nWhen to use:")
print("  - High cost of poor performance")
print("  - Want to exploit known good options")
print("  - Need conservative approach")

print("\n\n5. Thompson Sampling (High Confidence Prior)")
print("-" * 70)
print("Configuration:", thompson_confident_config)
print("\nCharacteristics:")
print("  - Both α and β high → lower variance in sampling")
print("  - More stable, less random exploration")
print("  - Good for: Have prior knowledge, want stability")
print("\nWhen to use:")
print("  - Have prior knowledge about arm quality")
print("  - Want more predictable behavior")
print("  - Reduce randomness in selection")

print("\n\n" + "="*70)
print("Summary Decision Guide")
print("="*70)

print("\nUse UCB if:")
print("  ✓ Need deterministic, reproducible results")
print("  ✓ Want explicit control over exploration (tune ucb_alpha)")
print("  ✓ Have stable, low-noise reward signals")
print("  ✓ Need theoretical regret guarantees")

print("\nUse Thompson Sampling if:")
print("  ✓ Want adaptive, intelligent exploration")
print("  ✓ Have noisy or stochastic reward signals")
print("  ✓ Prefer probabilistic decision-making")
print("  ✓ Want better empirical performance")
print("  ✓ Have prior knowledge (can set α, β priors)")

print("\n" + "="*70)
print("Parameter Tuning Tips")
print("="*70)

print("\nThompson Sampling Priors:")
print("  α=1, β=1    → Uniform (no bias)")
print("  α>β         → Optimistic (encourage exploration)")
print("  α<β         → Pessimistic (conservative)")
print("  α=β>1       → Low variance (stable)")

print("\nUCB Alpha:")
print("  α=0.5-2     → Typical range")
print("  α↑          → More exploration")
print("  α↓          → More exploitation")

print("\nNumber of Arms:")
print("  Few arms (2-4)    → Fast convergence, less granular")
print("  Many arms (8-16)  → Fine-grained control, more exploration")

print("\n" + "="*70)

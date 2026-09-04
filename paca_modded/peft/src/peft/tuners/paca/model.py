# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_PACA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.integrations import gather_params_ctx
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties

from .config import PacaConfig
from .layer import PacaLayer, dispatch_default

def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    kwargs["adapter_names"] = adapter_names
    return args, kwargs

class PacaUCBManager:
    def __init__(self, layers, num_arms, num_arms_to_select, ucb_alpha):
        self.layers = layers
        self.num_arms = num_arms
        self.num_arms_to_select = num_arms_to_select
        self.ucb_alpha = ucb_alpha
        self._initialize_vertical_arms()
        self.arm_rewards = torch.zeros(self.num_arms)
        self.arm_counts = torch.zeros(self.num_arms)
        self.total_epochs = torch.tensor(0.0)
        self.last_selected_arms: Optional[torch.Tensor] = None

    def _initialize_vertical_arms(self):
        per = len(self.layers)//self.num_arms
        self.arm_layers = []
        for i in range(self.num_arms):
            start = i*per
            end = (i+1)*per if i<self.num_arms-1 else len(self.layers)
            self.arm_layers.append(self.layers[start:end])

    def reselect_ucb_indices(self, adapter_name):
        t = self.total_epochs.item() + 1
        scores = torch.zeros(self.num_arms)
        for i in range(self.num_arms):
            n_i = self.arm_counts[i].item()
            mu = (self.arm_rewards[i]/n_i).item() if n_i>0 else 0.0
            explore = self.ucb_alpha*math.sqrt(2*math.log(t)/n_i) if n_i>0 else float('inf')
            scores[i] = mu + explore
        sel = torch.topk(scores, self.num_arms_to_select).indices
        self.last_selected_arms = sel.clone()
        # enable or disable arms
        for i, arm in enumerate(self.arm_layers):
            for layer in arm:
                if i in sel:
                    layer.enable_adapters(True)
                    layer.reselect_indices(adapter_name)
                else:
                    layer.enable_adapters(False)
        self.arm_counts[sel] += 1
        self.total_epochs = torch.tensor(t)

    def update_ucb_rewards(self, reward):
        if self.last_selected_arms is None:
            return
        for i in self.last_selected_arms:
            self.arm_rewards[i] += reward

    def print_logs(self):
        print("[PacaUCBManager - Vertical UCB Logs]")
        print(f"  Arm Rewards: {self.arm_rewards.tolist()}")
        print(f"  Arm Counts: {self.arm_counts.tolist()}")
        print(f"  Total Epochs: {int(self.total_epochs.item())}")

class PacaThompsonManager:
    """Vertical distribution Thompson Sampling manager"""
    def __init__(self, layers, num_arms, num_arms_to_select, alpha_prior=1.0, beta_prior=1.0):
        self.layers = layers
        self.num_arms = num_arms
        self.num_arms_to_select = num_arms_to_select
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self._initialize_vertical_arms()
        
        # Thompson Sampling parameters (Beta distribution)
        self.arm_successes = torch.full((self.num_arms,), alpha_prior, dtype=torch.float32)
        self.arm_failures = torch.full((self.num_arms,), beta_prior, dtype=torch.float32)
        self.arm_counts = torch.zeros(self.num_arms)
        self.total_epochs = torch.tensor(0.0)
        self.last_selected_arms: Optional[torch.Tensor] = None

    def _initialize_vertical_arms(self):
        per = len(self.layers) // self.num_arms
        self.arm_layers = []
        for i in range(self.num_arms):
            start = i * per
            end = (i + 1) * per if i < self.num_arms - 1 else len(self.layers)
            self.arm_layers.append(self.layers[start:end])

    def reselect_thompson_indices(self, adapter_name):
        """Thompson Sampling for vertical distribution"""
        t = self.total_epochs.item() + 1
        
        # Sample from Beta distribution for each arm
        samples = torch.zeros(self.num_arms)
        for i in range(self.num_arms):
            alpha = self.arm_successes[i].item()
            beta = self.arm_failures[i].item()
            samples[i] = torch.distributions.Beta(alpha, beta).sample()
        
        # Select top arms
        sel = torch.topk(samples, self.num_arms_to_select).indices
        self.last_selected_arms = sel.clone()
        
        # Enable/disable arms
        for i, arm in enumerate(self.arm_layers):
            for layer in arm:
                if i in sel:
                    layer.enable_adapters(True)
                    layer.reselect_indices(adapter_name)
                else:
                    layer.enable_adapters(False)
        
        self.arm_counts[sel] += 1
        self.total_epochs = torch.tensor(t)
        self._last_samples = samples.clone()

    def update_thompson_rewards(self, reward):
        """Update Beta distribution parameters"""
        if self.last_selected_arms is None:
            return
        
        for i in self.last_selected_arms:
            if reward > 0:
                self.arm_successes[i] += abs(reward)
            else:
                self.arm_failures[i] += abs(reward)

    def print_logs(self):
        print("[PacaThompsonManager - Vertical Thompson Sampling Logs]")
        print(f"  Arm Successes (Alpha): {self.arm_successes.tolist()}")
        print(f"  Arm Failures (Beta): {self.arm_failures.tolist()}")
        print(f"  Arm Counts: {self.arm_counts.tolist()}")
        print(f"  Total Epochs: {int(self.total_epochs.item())}")
        if hasattr(self, '_last_samples'):
            print(f"  Last Sampled Values: {self._last_samples.tolist()}")

class PacaThompsonGaussianManager:
    """Vertical distribution Gaussian Thompson Sampling manager"""
    def __init__(self, layers, num_arms, num_arms_to_select, prior_mean=0.0, prior_std=1.0, noise_std=1.0):
        self.layers = layers
        self.num_arms = num_arms
        self.num_arms_to_select = num_arms_to_select
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.noise_std = noise_std
        self._initialize_vertical_arms()
        
        # Gaussian Thompson Sampling parameters
        self.arm_means = torch.full((self.num_arms,), prior_mean, dtype=torch.float32)
        self.arm_stds = torch.full((self.num_arms,), prior_std, dtype=torch.float32)
        self.arm_observation_counts = torch.zeros(self.num_arms, dtype=torch.long)
        self.arm_counts = torch.zeros(self.num_arms)
        self.total_epochs = torch.tensor(0.0)
        self.last_selected_arms: Optional[torch.Tensor] = None

    def _initialize_vertical_arms(self):
        per = len(self.layers) // self.num_arms
        self.arm_layers = []
        for i in range(self.num_arms):
            start = i * per
            end = (i + 1) * per if i < self.num_arms - 1 else len(self.layers)
            self.arm_layers.append(self.layers[start:end])

    def reselect_thompson_gaussian_indices(self, adapter_name):
        """Gaussian Thompson Sampling for vertical distribution"""
        t = self.total_epochs.item() + 1
        
        # Sample from Normal distribution for each arm
        samples = torch.zeros(self.num_arms)
        for i in range(self.num_arms):
            mean = self.arm_means[i].item()
            std = self.arm_stds[i].item()
            samples[i] = torch.distributions.Normal(mean, std).sample()
        
        # Select top arms
        sel = torch.topk(samples, self.num_arms_to_select).indices
        self.last_selected_arms = sel.clone()
        
        # Enable/disable arms
        for i, arm in enumerate(self.arm_layers):
            for layer in arm:
                if i in sel:
                    layer.enable_adapters(True)
                    layer.reselect_indices(adapter_name)
                else:
                    layer.enable_adapters(False)
        
        self.arm_counts[sel] += 1
        self.total_epochs = torch.tensor(t)
        self._last_samples = samples.clone()

    def update_thompson_gaussian_rewards(self, reward):
        """Update Gaussian parameters using Bayesian updating"""
        if self.last_selected_arms is None:
            return
        
        for i in self.last_selected_arms:
            prior_mean = self.arm_means[i].item()
            prior_var = self.arm_stds[i].item() ** 2
            obs_var = self.noise_std ** 2
            
            # Bayesian update
            posterior_precision = 1.0 / prior_var + 1.0 / obs_var
            posterior_var = 1.0 / posterior_precision
            posterior_mean = posterior_var * (prior_mean / prior_var + reward / obs_var)
            
            self.arm_means[i] = posterior_mean
            self.arm_stds[i] = torch.sqrt(torch.tensor(posterior_var))
            self.arm_observation_counts[i] += 1

    def print_logs(self):
        print("[PacaThompsonGaussianManager - Vertical Gaussian Thompson Sampling Logs]")
        print(f"  Arm Means: {self.arm_means.tolist()}")
        print(f"  Arm Stds: {self.arm_stds.tolist()}")
        print(f"  Arm Observation Counts: {self.arm_observation_counts.tolist()}")
        print(f"  Arm Counts: {self.arm_counts.tolist()}")
        print(f"  Total Epochs: {int(self.total_epochs.item())}")
        if hasattr(self, '_last_samples'):
            print(f"  Last Sampled Values: {self._last_samples.tolist()}")

class PacaGradientChainManager:
    """Gradient-based chain selection manager"""
    def __init__(self, model, config, adapter_name, device):
        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.device = device
        self.num_arms = config.num_arms
        self.num_arms_to_select = config.num_arms_to_select
        self.warmup_epochs = config.gradient_chain_warmup_epochs
        self.num_samples = config.gradient_chain_num_samples
        
        # Get all PaCA layers in order
        self.paca_layers = self._get_paca_layers_ordered()
        self.num_layers = len(self.paca_layers)
        
        # Chain storage: List of dicts, each dict maps layer_name -> column_index
        self.chains = []
        self.chains_built = False
        self.current_epoch = 0
        
        # Arm selection method (UCB or Thompson)
        self.arm_selection_method = config.arm_selection_method
        self.thompson_distribution = config.thompson_distribution
        
        # UCB/Thompson parameters
        if self.arm_selection_method == "thompson":
            if self.thompson_distribution == "gaussian":
                # Gaussian Thompson Sampling
                self.arm_means = torch.full((self.num_arms,), config.thompson_gaussian_prior_mean, dtype=torch.float32, device=device)
                self.arm_stds = torch.full((self.num_arms,), config.thompson_gaussian_prior_std, dtype=torch.float32, device=device)
                self.arm_observation_counts = torch.zeros(self.num_arms, dtype=torch.long, device=device)
                self.noise_std = config.thompson_gaussian_noise_std
            else:
                # Beta Thompson Sampling
                self.arm_successes = torch.full((self.num_arms,), config.thompson_alpha_prior, dtype=torch.float32, device=device)
                self.arm_failures = torch.full((self.num_arms,), config.thompson_beta_prior, dtype=torch.float32, device=device)
        else:
            # UCB
            self.arm_rewards = torch.zeros(self.num_arms, device=device)
            self.arm_counts = torch.zeros(self.num_arms, device=device)
            self.ucb_alpha = config.ucb_alpha
        
        self.total_epochs = torch.tensor(0.0, device=device)
        self.last_selected_arms = None
        
    def _get_paca_layers_ordered(self):
        """Get PaCA layers in order (skip first layer, start from 2nd)"""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, PacaLayer) and self.adapter_name in module.paca_w:
                layers.append((name, module))
        
        # Skip first layer (input layer), start from second layer
        if len(layers) > 1:
            layers = layers[1:]  # Remove first layer
        
        return layers
    
    def build_gradient_chains(self, train_loader):
        """
        Build gradient chains by following max-gradient paths through network.
        
        For each starting neuron in the 2nd layer:
        1. Compute ∂(activation)/∂(weight) for all weights in that layer
        2. Select weight with maximum gradient (determines next neuron)
        3. Repeat for subsequent layers
        """
        print("\n" + "=" * 60)
        print("Building Gradient Chains...")
        print("=" * 60)
        
        self.model.eval()  # Set to eval mode for consistent gradients
        
        # Accumulate gradients over multiple batches
        accumulated_grads = {name: [] for name, _ in self.paca_layers}
        
        print(f"Sampling {self.num_samples} batches to estimate gradients...")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= self.num_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Backward pass to compute gradients
            # We need gradients w.r.t. activations, so we'll use the outputs
            loss = outputs.mean()  # Dummy loss to trigger backward
            loss.backward()
            
            # Collect gradients of paca_w for each layer
            for layer_name, layer_module in self.paca_layers:
                if layer_module.paca_w[self.adapter_name].grad is not None:
                    # Gradient shape: [out_features, r]
                    grad = layer_module.paca_w[self.adapter_name].grad.abs().clone()
                    accumulated_grads[layer_name].append(grad)
            
            print(f"  Batch {batch_idx + 1}/{self.num_samples} processed", end='\r')
        
        print("\n\nAveraging gradients across batches...")
        
        # Average gradients across batches
        avg_grads = {}
        for layer_name, grads_list in accumulated_grads.items():
            if len(grads_list) > 0:
                avg_grads[layer_name] = torch.stack(grads_list).mean(dim=0)
        
        # Now build chains
        print(f"\nBuilding {self.num_arms} gradient chains...")
        
        # Get first layer info
        first_layer_name, first_layer_module = self.paca_layers[0]
        num_start_neurons = first_layer_module.r[self.adapter_name]  # Number of possible starting points
        
        # Sample starting neurons if we have more than num_arms
        if num_start_neurons > self.num_arms:
            start_neurons = torch.randperm(num_start_neurons, device=self.device)[:self.num_arms].tolist()
        else:
            start_neurons = list(range(min(num_start_neurons, self.num_arms)))
        
        self.chains = []
        
        for arm_idx, start_neuron_idx in enumerate(start_neurons):
            chain = {}
            current_neuron_idx = start_neuron_idx
            
            print(f"\nChain {arm_idx}: Starting from neuron {start_neuron_idx}")
            
            for layer_idx, (layer_name, layer_module) in enumerate(self.paca_layers):
                if layer_name not in avg_grads:
                    print(f"  Warning: No gradients for layer {layer_name}, skipping")
                    continue
                
                grad_matrix = avg_grads[layer_name]  # Shape: [out_features, r]
                
                # Get gradients for weights connected FROM current_neuron_idx
                # In PaCA, paca_w selects input columns, so we look at column current_neuron_idx
                if current_neuron_idx < grad_matrix.shape[1]:
                    neuron_grads = grad_matrix[:, current_neuron_idx]  # Shape: [out_features]
                    
                    # Select output neuron with maximum gradient
                    next_neuron_idx = neuron_grads.argmax().item()
                    
                    # Store this connection in the chain
                    chain[layer_name] = current_neuron_idx
                    
                    print(f"  Layer {layer_idx} ({layer_name}): input_col={current_neuron_idx} -> output_neuron={next_neuron_idx} (grad={neuron_grads[next_neuron_idx]:.4f})")
                    
                    # Move to next layer with selected neuron
                    current_neuron_idx = next_neuron_idx % grad_matrix.shape[1]  # Wrap around if needed
                else:
                    # Fallback if index out of range
                    chain[layer_name] = 0
                    current_neuron_idx = 0
                    print(f"  Layer {layer_idx} ({layer_name}): Index out of range, using 0")
            
            self.chains.append(chain)
        
        self.chains_built = True
        print("\n" + "=" * 60)
        print(f"✓ Successfully built {len(self.chains)} gradient chains")
        print("=" * 60)
        
        # Set model back to training mode
        self.model.train()
    
    def reselect_indices(self):
        """Select arms using UCB or Thompson Sampling and apply chain indices"""
        if not self.chains_built:
            print("Warning: Chains not built yet, skipping selection")
            return
        
        t = self.total_epochs.item() + 1
        
        # Select arms based on method
        if self.arm_selection_method == "thompson":
            if self.thompson_distribution == "gaussian":
                # Gaussian Thompson Sampling
                samples = torch.zeros(self.num_arms, device=self.device)
                for i in range(self.num_arms):
                    mean = self.arm_means[i].item()
                    std = self.arm_stds[i].item()
                    samples[i] = torch.distributions.Normal(mean, std).sample()
                sel = torch.topk(samples, self.num_arms_to_select).indices
                self._last_samples = samples.clone()
            else:
                # Beta Thompson Sampling
                samples = torch.zeros(self.num_arms, device=self.device)
                for i in range(self.num_arms):
                    alpha = self.arm_successes[i].item()
                    beta = self.arm_failures[i].item()
                    samples[i] = torch.distributions.Beta(alpha, beta).sample()
                sel = torch.topk(samples, self.num_arms_to_select).indices
                self._last_samples = samples.clone()
        else:
            # UCB
            scores = torch.zeros(self.num_arms, device=self.device)
            for i in range(self.num_arms):
                n_i = self.arm_counts[i].item()
                mu = (self.arm_rewards[i] / n_i).item() if n_i > 0 else 0.0
                explore = self.ucb_alpha * math.sqrt(2 * math.log(t) / n_i) if n_i > 0 else float('inf')
                scores[i] = mu + explore
            sel = torch.topk(scores, self.num_arms_to_select).indices
            self._last_ucb_scores = scores.clone()
        
        self.last_selected_arms = sel.clone()
        
        # Apply chain indices to each selected arm
        for layer_name, layer_module in self.paca_layers:
            # Collect indices from selected chains for this layer
            selected_indices = []
            for arm_idx in sel:
                if arm_idx < len(self.chains):
                    chain = self.chains[arm_idx]
                    if layer_name in chain:
                        selected_indices.append(chain[layer_name])
            
            if len(selected_indices) > 0:
                # Set paca_indices to selected chain indices
                indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)
                
                # Ensure we have exactly r indices (pad or trim if needed)
                r = layer_module.r[self.adapter_name]
                if len(selected_indices) < r:
                    # Pad with random indices
                    extra_needed = r - len(selected_indices)
                    extra_indices = torch.randperm(layer_module.in_features, device=self.device)[:extra_needed]
                    indices_tensor = torch.cat([indices_tensor, extra_indices])
                elif len(selected_indices) > r:
                    # Trim
                    indices_tensor = indices_tensor[:r]
                
                # Update layer's paca_indices
                layer_module.paca_indices[self.adapter_name] = indices_tensor
                layer_module.paca_w[self.adapter_name].data.zero_()
                
                # Reinitialize paca_w with new indices
                with gather_params_ctx(layer_module.get_base_layer().weight):
                    layer_module.paca_init(self.adapter_name)
        
        # Update counts
        if self.arm_selection_method == "thompson":
            pass  # Counts updated in reward function
        else:
            self.arm_counts[sel] += 1
        
        self.total_epochs = torch.tensor(t, device=self.device)
    
    def update_rewards(self, reward):
        """Update arm statistics based on reward"""
        if self.last_selected_arms is None:
            return
        
        for i in self.last_selected_arms:
            if self.arm_selection_method == "thompson":
                if self.thompson_distribution == "gaussian":
                    # Gaussian Bayesian update
                    prior_mean = self.arm_means[i].item()
                    prior_var = self.arm_stds[i].item() ** 2
                    obs_var = self.noise_std ** 2
                    
                    posterior_precision = 1.0 / prior_var + 1.0 / obs_var
                    posterior_var = 1.0 / posterior_precision
                    posterior_mean = posterior_var * (prior_mean / prior_var + reward / obs_var)
                    
                    self.arm_means[i] = posterior_mean
                    self.arm_stds[i] = torch.sqrt(torch.tensor(posterior_var))
                    self.arm_observation_counts[i] += 1
                else:
                    # Beta update
                    if reward > 0:
                        self.arm_successes[i] += abs(reward)
                    else:
                        self.arm_failures[i] += abs(reward)
            else:
                # UCB update
                self.arm_rewards[i] += reward
    
    def print_logs(self):
        """Print gradient chain statistics"""
        print("[PacaGradientChainManager - Gradient Chain Logs]")
        print(f"  Chains Built: {self.chains_built}")
        print(f"  Number of Chains: {len(self.chains)}")
        print(f"  Current Epoch: {int(self.total_epochs.item())}")
        
        if self.last_selected_arms is not None:
            print(f"  Selected Arms: {self.last_selected_arms.tolist()}")
        
        if self.arm_selection_method == "thompson":
            if self.thompson_distribution == "gaussian":
                print(f"  Arm Means: {self.arm_means.tolist()}")
                print(f"  Arm Stds: {self.arm_stds.tolist()}")
                print(f"  Arm Observation Counts: {self.arm_observation_counts.tolist()}")
                if hasattr(self, '_last_samples'):
                    print(f"  Last Sampled Values: {self._last_samples.tolist()}")
            else:
                print(f"  Arm Successes (Alpha): {self.arm_successes.tolist()}")
                print(f"  Arm Failures (Beta): {self.arm_failures.tolist()}")
                if hasattr(self, '_last_samples'):
                    print(f"  Last Sampled Values: {self._last_samples.tolist()}")
        else:
            print(f"  Arm Rewards: {self.arm_rewards.tolist()}")
            print(f"  Arm Counts: {self.arm_counts.tolist()}")
            if hasattr(self, '_last_ucb_scores'):
                print(f"  Last UCB Scores: {self._last_ucb_scores.tolist()}")

class PacaModel(BaseTuner):
    prefix: str = "paca_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        self.ucb_manager = None
        self.thompson_manager = None
        self.thompson_gaussian_manager = None
        self.gradient_chain_manager = None
        
        # Extract the actual config object (might be passed as dict)
        paca_config = config[adapter_name] if isinstance(config, dict) else config
        
        if paca_config.arm_distribution == "gradient_chain":
            # Gradient chain mode - manager will be created, chains built after warmup
            device = next(model.parameters()).device
            self.gradient_chain_manager = PacaGradientChainManager(
                model, paca_config, adapter_name, device
            )
        elif paca_config.arm_distribution == "vertical":
            paca_layers = [module for module in self.model.modules() if isinstance(module, PacaLayer)]
            
            if paca_config.arm_selection_method == "thompson":
                if paca_config.thompson_distribution == "gaussian":
                    self.thompson_gaussian_manager = PacaThompsonGaussianManager(
                        paca_layers,
                        paca_config.num_arms,
                        paca_config.num_arms_to_select,
                        prior_mean=paca_config.thompson_gaussian_prior_mean,
                        prior_std=paca_config.thompson_gaussian_prior_std,
                        noise_std=paca_config.thompson_gaussian_noise_std,
                    )
                else:
                    self.thompson_manager = PacaThompsonManager(
                        paca_layers,
                        paca_config.num_arms,
                        paca_config.num_arms_to_select,
                        alpha_prior=paca_config.thompson_alpha_prior,
                        beta_prior=paca_config.thompson_beta_prior,
                    )
            else:
                self.ucb_manager = PacaUCBManager(
                    paca_layers,
                    paca_config.num_arms,
                    paca_config.num_arms_to_select,
                    paca_config.ucb_alpha,
                )
            paca_layers = [module for module in self.model.modules() if isinstance(module, PacaLayer)]
            
            if paca_config.arm_selection_method == "thompson":
                if paca_config.thompson_distribution == "gaussian":
                    self.thompson_gaussian_manager = PacaThompsonGaussianManager(
                        paca_layers,
                        paca_config.num_arms,
                        paca_config.num_arms_to_select,
                        paca_config.thompson_gaussian_prior_mean,
                        paca_config.thompson_gaussian_prior_std,
                        paca_config.thompson_gaussian_noise_std
                    )
                else:
                    self.thompson_manager = PacaThompsonManager(
                        paca_layers, 
                        paca_config.num_arms, 
                        paca_config.num_arms_to_select,
                        paca_config.thompson_alpha_prior,
                        paca_config.thompson_beta_prior
                    )
            else:
                self.ucb_manager = PacaUCBManager(
                    paca_layers, 
                    paca_config.num_arms, 
                    paca_config.num_arms_to_select, 
                    paca_config.ucb_alpha
                )

    def _check_new_adapter_config(self, config: PacaConfig) -> None:
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(paca_config, key):
        return check_target_module_exists(paca_config, key)

    def _prepare_model(self, peft_config: PacaConfig, model: nn.Module):
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        paca_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        pattern_keys = list(chain(paca_config.rank_pattern.keys(), paca_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = paca_config.rank_pattern.get(target_name_key, paca_config.r)
        alpha = paca_config.alpha_pattern.get(target_name_key, paca_config.paca_alpha)

        kwargs = {
            "r": r,
            "paca_alpha": alpha,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "steps_per_epoch": paca_config.steps_per_epoch,
            "arm_distribution": paca_config.arm_distribution,
            "num_arms": paca_config.num_arms,
            "num_arms_to_select": paca_config.num_arms_to_select,
            "ucb_alpha": paca_config.ucb_alpha,
            "arm_selection_method": paca_config.arm_selection_method,
            "thompson_distribution": paca_config.thompson_distribution,
            "thompson_alpha_prior": paca_config.thompson_alpha_prior,
            "thompson_beta_prior": paca_config.thompson_beta_prior,
            "thompson_gaussian_prior_mean": paca_config.thompson_gaussian_prior_mean,
            "thompson_gaussian_prior_std": paca_config.thompson_gaussian_prior_std,
            "thompson_gaussian_noise_std": paca_config.thompson_gaussian_noise_std,
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        if isinstance(target, PacaLayer):
            target.update_layer(
                adapter_name,
                r,
                paca_alpha=alpha,
            )
        else:
            new_module = self._create_new_module(paca_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")
    
    def print_logs(self, adapter_name: str = "default"):
        if self.gradient_chain_manager:
            # Gradient chain mode
            self.gradient_chain_manager.print_logs()
        else:
            # Horizontal or vertical mode
            for module in self.model.modules():
                if isinstance(module, PacaLayer):
                    if module.arm_distribution == "horizontal":
                        if adapter_name not in module.arm_rewards:
                            continue
                        sel_arms = torch.where(module.selected_arms[adapter_name] > 0)[0].tolist()
                        print(f"  Selected Arms: {sel_arms}")
                        
                        if module.arm_selection_method == "thompson":
                            if module.thompson_distribution == "gaussian":
                                print(f"  Arm Means: {module.arm_means[adapter_name].tolist()}")
                                print(f"  Arm Stds: {module.arm_stds[adapter_name].tolist()}")
                                print(f"  Arm Observation Counts: {module.arm_observation_counts[adapter_name].tolist()}")
                            else:
                                print(f"  Arm Successes (Alpha): {module.arm_successes[adapter_name].tolist()}")
                                print(f"  Arm Failures (Beta): {module.arm_failures[adapter_name].tolist()}")
                            if hasattr(module, "_last_thompson_samples"):
                                print(f"  Last Sampled Values: {module._last_thompson_samples.tolist()}")
                        else:
                            print(f"  Arm Rewards: {module.arm_rewards[adapter_name].tolist()}")
                            if hasattr(module, "_last_ucb_scores"):
                                print(f"  UCB Scores: {module._last_ucb_scores.tolist()}")
                        
                        print(f"  Arm Counts: {module.arm_counts[adapter_name].tolist()}")
                        print(f"  Value of t: {int(module.total_epochs[adapter_name].item())}")
                        break

            if self.thompson_gaussian_manager:
                self.thompson_gaussian_manager.print_logs()
            elif self.thompson_manager:
                self.thompson_manager.print_logs()
            elif self.ucb_manager:
                print("[PacaModel - Vertical UCB Logs]")
                self.ucb_manager.print_logs()

    def print_trainable_parameters(self) -> None:
        trainable_params = 0
        frozen_params = 0
        paca_trainable_params = 0

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            if name.startswith(self.prefix):
                if param.requires_grad:
                    trainable_params += num_params
                    paca_trainable_params += num_params
                else:
                    frozen_params += num_params
            else:
                if param.requires_grad:
                    trainable_params += num_params
                else:
                    frozen_params += num_params

        for module in self.model.modules():
            if isinstance(module, PacaLayer):
                for adapter_name in module.active_adapters:
                    if adapter_name in module.paca_w:
                        in_features = module.in_features
                        out_features = module.out_features
                        r = module.r[adapter_name]
                        frozen_params += (in_features - r) * out_features
                        frozen_params -= paca_trainable_params

        total_params = trainable_params + frozen_params
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")

    @staticmethod
    def _create_new_module(paca_config, adapter_name, target, **kwargs):
        dispatchers = []
        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit
            dispatchers.append(dispatch_bnb_4bit)
        dispatchers.append(dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, **kwargs)
            if new_module is not None:
                break

        if new_module is None:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )
        return new_module

    def reselect_paca_weights(self, adapter_name: str, train_loader=None) -> None:
        """
        Reselect PaCA weights based on arm distribution mode.
        
        Args:
            adapter_name: Name of the adapter
            train_loader: Training dataloader (required for gradient_chain mode during chain building)
        """
        if self.gradient_chain_manager:
            # Gradient chain mode
            self.gradient_chain_manager.current_epoch += 1
            
            # Build chains after warmup if not yet built
            if not self.gradient_chain_manager.chains_built:
                if self.gradient_chain_manager.current_epoch >= self.gradient_chain_manager.warmup_epochs:
                    if train_loader is None:
                        print("Warning: train_loader required for gradient chain building, skipping")
                        return
                    self.gradient_chain_manager.build_gradient_chains(train_loader)
                else:
                    print(f"Gradient chain warmup: {self.gradient_chain_manager.current_epoch}/{self.gradient_chain_manager.warmup_epochs}")
                    return
            
            # Reselect arms using chains
            self.gradient_chain_manager.reselect_indices()
        else:
            # Horizontal or vertical mode
            for module in self.model.modules():
                if isinstance(module, PacaLayer):
                    if adapter_name in module.paca_w:
                        if module.arm_distribution == "horizontal" and not self.peft_config[adapter_name].use_fixed_indices:
                            if module.arm_selection_method == "thompson":
                                if module.thompson_distribution == "gaussian":
                                    module.reselect_thompson_gaussian_indices(adapter_name)
                                else:
                                    module.reselect_thompson_indices(adapter_name)
                            else:
                                module.reselect_ucb_indices(adapter_name)
                        elif module.arm_distribution == "vertical":
                            if self.thompson_gaussian_manager:
                                self.thompson_gaussian_manager.reselect_thompson_gaussian_indices(adapter_name)
                            elif self.thompson_manager:
                                self.thompson_manager.reselect_thompson_indices(adapter_name)
                            elif self.ucb_manager:
                                self.ucb_manager.reselect_ucb_indices(adapter_name)

    def update_ucb_rewards(self, adapter_name: str, reward: float) -> None:
        """Updated to handle UCB, Thompson Sampling (Beta), Thompson Sampling (Gaussian), and Gradient Chains"""
        if self.gradient_chain_manager:
            # Gradient chain mode
            self.gradient_chain_manager.update_rewards(reward)
        else:
            # Horizontal or vertical mode
            for module in self.model.modules():
                if isinstance(module, PacaLayer):
                    if adapter_name in module.paca_w and module.arm_distribution == "horizontal":
                        if module.arm_selection_method == "thompson":
                            if module.thompson_distribution == "gaussian":
                                module.update_thompson_gaussian_rewards(adapter_name, reward)
                            else:
                                module.update_thompson_rewards(adapter_name, reward)
                        else:
                            module.update_ucb_rewards(adapter_name, reward)
            
            if self.thompson_gaussian_manager and self.peft_config[adapter_name].arm_distribution == "vertical":
                self.thompson_gaussian_manager.update_thompson_gaussian_rewards(reward)
            elif self.thompson_manager and self.peft_config[adapter_name].arm_distribution == "vertical":
                self.thompson_manager.update_thompson_rewards(reward)
            elif self.ucb_manager and self.peft_config[adapter_name].arm_distribution == "vertical":
                self.ucb_manager.update_ucb_rewards(reward)

    def get_best_indices(self, adapter_name: str = "default") -> dict:
        """
        Retrieves the most selected indices for each PacaLayer in horizontal mode.
        
        Args:
            adapter_name (str): Name of the adapter to query.
        
        Returns:
            dict: Mapping of layer names to their most selected indices.
        """
        best_indices = {}
        for name, module in self.model.named_modules():
            if isinstance(module, PacaLayer) and module.arm_distribution == "horizontal":
                try:
                    top_indices = module.get_most_selected_indices(adapter_name)
                    best_indices[name] = top_indices
                except ValueError:
                    pass  # Adapter not found in this layer
        return best_indices

    def set_fixed_indices(self, best_indices: dict, adapter_name: str = "default") -> None:
        """
        Sets fixed paca_indices for each PacaLayer to train only the most selected indices.
        
        Args:
            best_indices (dict): Mapping of layer names to their most selected indices.
            adapter_name (str): Name of the adapter to configure.
        """
        for name, indices in best_indices.items():
            try:
                module = self.model.get_submodule(name)
                if isinstance(module, PacaLayer):
                    device = module.get_base_layer().weight.device
                    module.paca_indices[adapter_name] = indices.to(device)
                    module.paca_init(adapter_name)
            except AttributeError:
                continue  # Skip if module not found

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        for module in self.model.modules():
            if isinstance(module, PacaLayer):
                if module.merged:
                    raise ValueError("Adapter cannot be set when the model is merged")
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is None:
            yield
            return

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        hook_handles = []
        for module in self.modules():
            if isinstance(module, PacaLayer):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge Paca layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge Paca layers when base model layers are replicated")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_PACA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_PACA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)
        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, PacaLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]
        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        return self._unload_and_optionally_merge(merge=False)
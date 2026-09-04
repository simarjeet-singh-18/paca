# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0sus (the "License");
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
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

from .config import PacaConfig
from .._buffer_dict import BufferDict
from torch.cuda.amp import custom_fwd, custom_bwd

class PacaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("paca_w",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "paca_alpha", "scaling")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.paca_alpha = {}
        self.scaling = {}
        self.paca_w = nn.ParameterDict({})
        self.paca_indices = BufferDict({}, persistent=True)
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
        # UCB attributes
        self.arm_distribution = kwargs.get("arm_distribution", "horizontal")
        self.num_arms = kwargs.get("num_arms", 8)
        self.num_arms_to_select = kwargs.get("num_arms_to_select", 2)
        self.ucb_alpha = kwargs.get("ucb_alpha", 0.5)
        self.arm_indices = BufferDict({}, persistent=True)
        self.arm_rewards = BufferDict({}, persistent=True)
        self.arm_counts = BufferDict({}, persistent=True)
        self.total_epochs = BufferDict({}, persistent=True)
        self.selected_arms = BufferDict({}, persistent=True)
        self.selection_counts = BufferDict({}, persistent=True)  # New: track individual index selections
        
        # Thompson Sampling attributes
        self.arm_selection_method = kwargs.get("arm_selection_method", "ucb")
        self.thompson_distribution = kwargs.get("thompson_distribution", "beta")
        self.thompson_alpha_prior = kwargs.get("thompson_alpha_prior", 1.0)
        self.thompson_beta_prior = kwargs.get("thompson_beta_prior", 1.0)
        self.arm_successes = BufferDict({}, persistent=True)  # Alpha parameter for Beta distribution
        self.arm_failures = BufferDict({}, persistent=True)   # Beta parameter for Beta distribution
        
        # Gaussian Thompson Sampling attributes
        self.thompson_gaussian_prior_mean = kwargs.get("thompson_gaussian_prior_mean", 0.0)
        self.thompson_gaussian_prior_std = kwargs.get("thompson_gaussian_prior_std", 1.0)
        self.thompson_gaussian_noise_std = kwargs.get("thompson_gaussian_noise_std", 1.0)
        self.arm_means = BufferDict({}, persistent=True)  # Posterior means for Gaussian
        self.arm_stds = BufferDict({}, persistent=True)   # Posterior stds for Gaussian
        self.arm_observation_counts = BufferDict({}, persistent=True)  # Number of observations for Gaussian

        base = self.get_base_layer()
        if isinstance(base, nn.Linear):
            self.in_features, self.out_features = base.in_features, base.out_features
        elif isinstance(base, Conv1D):
            self.in_features, self.out_features = base.weight.shape

        if self.arm_distribution == "horizontal":
            self._initialize_horizontal_arms()

    def _initialize_horizontal_arms(self):
        dev = self.get_base_layer().weight.device
        for name in self.r:
            indices = torch.randperm(self.in_features, device=dev)
            splits = torch.tensor_split(indices, self.num_arms)
            max_len = max(len(split) for split in splits)
            tensor = torch.full((self.num_arms, max_len), -1, dtype=torch.long, device=dev)
            for i, split in enumerate(splits):
                tensor[i, :len(split)] = split
            self.arm_indices[name] = tensor
            self.arm_rewards[name] = torch.zeros(self.num_arms, device=dev)
            self.arm_counts[name] = torch.ones(self.num_arms, device=dev)
            self.total_epochs[name] = torch.tensor(1.0, device=dev)
            self.selected_arms[name] = torch.zeros(self.num_arms, dtype=torch.long, device=dev)
            self.selection_counts[name] = torch.zeros(self.in_features, device=dev, dtype=torch.long)
            
            # Thompson Sampling initialization
            self.arm_successes[name] = torch.full(
                (self.num_arms,), self.thompson_alpha_prior, device=dev, dtype=torch.float32
            )
            self.arm_failures[name] = torch.full(
                (self.num_arms,), self.thompson_beta_prior, device=dev, dtype=torch.float32
            )
            
            # Gaussian Thompson Sampling initialization
            self.arm_means[name] = torch.full(
                (self.num_arms,), self.thompson_gaussian_prior_mean, device=dev, dtype=torch.float32
            )
            self.arm_stds[name] = torch.full(
                (self.num_arms,), self.thompson_gaussian_prior_std, device=dev, dtype=torch.float32
            )
            self.arm_observation_counts[name] = torch.zeros(self.num_arms, device=dev, dtype=torch.long)

    def update_layer(self, adapter_name, r, paca_alpha):
        if r <= 0:
            raise ValueError("r must be positive")
        self.r[adapter_name] = r
        self.paca_alpha[adapter_name] = paca_alpha
        self.paca_w[adapter_name] = nn.Parameter(torch.zeros(self.out_features, r))
        self.scaling[adapter_name] = paca_alpha / r
        # Initialize horizontal arms BEFORE reselecting indices
        if self.arm_distribution == "horizontal":
            self._initialize_horizontal_arms()
        self.reselect_indices(adapter_name)
        with gather_params_ctx(self.get_base_layer().weight):
            self.paca_init(adapter_name)

    def reselect_indices(self, adapter_name):
        """Route to appropriate selection method"""
        # Skip layer-level selection for gradient chains (handled at model level)
        if self.arm_distribution == "gradient_chain":
            dev = self.get_base_layer().weight.device
            self.paca_indices[adapter_name] = torch.randperm(self.in_features, device=dev)[:self.r[adapter_name]]
            self.paca_w[adapter_name].data.zero_()
            with gather_params_ctx(self.get_base_layer().weight):
                self.paca_init(adapter_name)
            return
        
        if self.arm_selection_method == "thompson":
            if self.thompson_distribution == "gaussian":
                self.reselect_thompson_gaussian_indices(adapter_name)
            else:
                self.reselect_thompson_indices(adapter_name)
        else:
            dev = self.get_base_layer().weight.device
            self.paca_indices[adapter_name] = torch.randperm(self.in_features, device=dev)[:self.r[adapter_name]]
            self.paca_w[adapter_name].data.zero_()
            with gather_params_ctx(self.get_base_layer().weight):
                self.paca_init(adapter_name)

    def reselect_ucb_indices(self, adapter_name):
        """Route to appropriate selection method"""
        # Skip layer-level selection for gradient chains (handled at model level)
        if self.arm_distribution == "gradient_chain":
            return
        
        if self.arm_selection_method == "thompson":
            if self.thompson_distribution == "gaussian":
                self.reselect_thompson_gaussian_indices(adapter_name)
            else:
                self.reselect_thompson_indices(adapter_name)
            return
        
        dev = self.get_base_layer().weight.device
        t = self.total_epochs[adapter_name].item() + 1
        scores = torch.zeros(self.num_arms, device=dev)
        for i in range(self.num_arms):
            n_i = self.arm_counts[adapter_name][i].item()
            mu = (self.arm_rewards[adapter_name][i] / n_i).item() if n_i>0 else 0.0
            explore = self.ucb_alpha * math.sqrt(2*math.log(t)/n_i) if n_i>0 else float('inf')
            scores[i] = mu + explore
        sel = torch.topk(scores, self.num_arms_to_select).indices
        # store selected arms
        buf = torch.zeros(self.num_arms, dtype=torch.long, device=dev)
        buf[sel] = 1
        self.selected_arms[adapter_name] = buf
        # collect indices from arms
        parts = [self.arm_indices[adapter_name][i][self.arm_indices[adapter_name][i]!=-1] for i in sel]
        idx = torch.cat(parts)
        # adjust to r
        if idx.numel()>self.r[adapter_name]: idx=idx[:self.r[adapter_name]]
        elif idx.numel()<self.r[adapter_name]:
            extra = torch.randperm(self.in_features, device=dev)[:self.r[adapter_name]-idx.numel()]
            idx = torch.cat([idx, extra])
        self.paca_indices[adapter_name] = idx
        self.paca_w[adapter_name].data.zero_()
        with gather_params_ctx(self.get_base_layer().weight):
            self.paca_init(adapter_name)
        self.arm_counts[adapter_name][sel] += 1
        self.total_epochs[adapter_name] = torch.tensor(t, device=dev)
        # Update selection counts for individual indices
        self.selection_counts[adapter_name][idx] += 1
        self._last_ucb_scores = scores.clone()

    def update_ucb_rewards(self, adapter_name, reward):
        sel = torch.where(self.selected_arms[adapter_name]>0)[0]
        for i in sel:
            self.arm_rewards[adapter_name][i] += reward

    def reselect_thompson_indices(self, adapter_name):
        """
        Thompson Sampling based arm selection.
        Samples from Beta distribution for each arm and selects top-k.
        """
        dev = self.get_base_layer().weight.device
        
        # Sample from Beta distribution for each arm
        samples = torch.zeros(self.num_arms, device=dev)
        for i in range(self.num_arms):
            alpha = self.arm_successes[adapter_name][i].item()
            beta = self.arm_failures[adapter_name][i].item()
            
            # Sample from Beta(alpha, beta)
            samples[i] = torch.distributions.Beta(alpha, beta).sample()
        
        # Select top num_arms_to_select arms based on sampled values
        sel = torch.topk(samples, self.num_arms_to_select).indices
        
        # Store selected arms
        buf = torch.zeros(self.num_arms, dtype=torch.long, device=dev)
        buf[sel] = 1
        self.selected_arms[adapter_name] = buf
        
        # Collect indices from selected arms
        parts = [self.arm_indices[adapter_name][i][self.arm_indices[adapter_name][i]!=-1] for i in sel]
        idx = torch.cat(parts)
        
        # Adjust to r
        if idx.numel() > self.r[adapter_name]:
            idx = idx[:self.r[adapter_name]]
        elif idx.numel() < self.r[adapter_name]:
            extra = torch.randperm(self.in_features, device=dev)[:self.r[adapter_name]-idx.numel()]
            idx = torch.cat([idx, extra])
        
        self.paca_indices[adapter_name] = idx
        self.paca_w[adapter_name].data.zero_()
        
        with gather_params_ctx(self.get_base_layer().weight):
            self.paca_init(adapter_name)
        
        # Update counts
        self.arm_counts[adapter_name][sel] += 1
        t = self.total_epochs[adapter_name].item() + 1
        self.total_epochs[adapter_name] = torch.tensor(t, device=dev)
        
        # Update selection counts for individual indices
        self.selection_counts[adapter_name][idx] += 1
        
        # Store samples for logging
        self._last_thompson_samples = samples.clone()

    def update_thompson_rewards(self, adapter_name, reward):
        """
        Update Thompson Sampling parameters based on reward.
        Positive rewards increase successes (alpha), negative increase failures (beta).
        """
        sel = torch.where(self.selected_arms[adapter_name] > 0)[0]
        
        for i in sel:
            if reward > 0:
                # Success: increase alpha
                self.arm_successes[adapter_name][i] += abs(reward)
            else:
                # Failure: increase beta
                self.arm_failures[adapter_name][i] += abs(reward)

    def reselect_thompson_gaussian_indices(self, adapter_name):
        """
        Thompson Sampling with Gaussian/Normal distribution.
        Uses Bayesian updating for posterior mean and variance.
        """
        dev = self.get_base_layer().weight.device
        
        # Sample from Gaussian distribution for each arm
        samples = torch.zeros(self.num_arms, device=dev)
        for i in range(self.num_arms):
            mean = self.arm_means[adapter_name][i].item()
            std = self.arm_stds[adapter_name][i].item()
            
            # Sample from Normal(mean, std)
            samples[i] = torch.distributions.Normal(mean, std).sample()
        
        # Select top num_arms_to_select arms based on sampled values
        sel = torch.topk(samples, self.num_arms_to_select).indices
        
        # Store selected arms
        buf = torch.zeros(self.num_arms, dtype=torch.long, device=dev)
        buf[sel] = 1
        self.selected_arms[adapter_name] = buf
        
        # Collect indices from selected arms
        parts = [self.arm_indices[adapter_name][i][self.arm_indices[adapter_name][i]!=-1] for i in sel]
        idx = torch.cat(parts)
        
        # Adjust to r
        if idx.numel() > self.r[adapter_name]:
            idx = idx[:self.r[adapter_name]]
        elif idx.numel() < self.r[adapter_name]:
            extra = torch.randperm(self.in_features, device=dev)[:self.r[adapter_name]-idx.numel()]
            idx = torch.cat([idx, extra])
        
        self.paca_indices[adapter_name] = idx
        self.paca_w[adapter_name].data.zero_()
        
        with gather_params_ctx(self.get_base_layer().weight):
            self.paca_init(adapter_name)
        
        # Update counts
        self.arm_counts[adapter_name][sel] += 1
        t = self.total_epochs[adapter_name].item() + 1
        self.total_epochs[adapter_name] = torch.tensor(t, device=dev)
        
        # Update selection counts for individual indices
        self.selection_counts[adapter_name][idx] += 1
        
        # Store samples for logging
        self._last_thompson_samples = samples.clone()

    def update_thompson_gaussian_rewards(self, adapter_name, reward):
        """
        Update Gaussian Thompson Sampling parameters using Bayesian updating.
        Updates posterior mean and variance based on observed reward.
        """
        sel = torch.where(self.selected_arms[adapter_name] > 0)[0]
        
        for i in sel:
            n = self.arm_observation_counts[adapter_name][i].item()
            prior_mean = self.arm_means[adapter_name][i].item()
            prior_var = self.arm_stds[adapter_name][i].item() ** 2
            obs_var = self.thompson_gaussian_noise_std ** 2
            
            # Bayesian update for Gaussian with known variance
            # Posterior precision = prior precision + observation precision
            posterior_precision = 1.0 / prior_var + 1.0 / obs_var
            posterior_var = 1.0 / posterior_precision
            
            # Posterior mean = weighted average of prior mean and observation
            posterior_mean = posterior_var * (prior_mean / prior_var + reward / obs_var)
            
            # Update stored values
            self.arm_means[adapter_name][i] = posterior_mean
            self.arm_stds[adapter_name][i] = torch.sqrt(torch.tensor(posterior_var))
            self.arm_observation_counts[adapter_name][i] += 1

    def get_most_selected_indices(self, adapter_name):
        """
        Retrieves the indices that were most frequently selected during training.
        
        Args:
            adapter_name (str): Name of the adapter to query.
        
        Returns:
            torch.Tensor: The top r indices with the highest selection counts.
        """
        if adapter_name not in self.selection_counts:
            raise ValueError(f"Adapter {adapter_name} not found")
        counts = self.selection_counts[adapter_name]
        _, top_indices = torch.topk(counts, self.r[adapter_name])
        return top_indices

    def paca_init(self, adapter_name):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if adapter_name not in self.scaling:
            raise ValueError(f"Scaling factor for adapter {adapter_name} not found")
        self.paca_w[adapter_name].data = prune_weight(weight, self.paca_indices[adapter_name]) / self.scaling[adapter_name]
        
    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            return
        self.scaling[adapter] = scale * self.paca_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_w.keys():
                continue
            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_w.keys():
                continue
            if scale is None:
                self.scaling[active_adapter] = self.paca_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return
        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)
        if self.merged:
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)
        unique_adapters = set(self.active_adapters)
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

class Linear(nn.Module, PacaLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        paca_alpha: int = 1,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        gradient_accumulation_steps: int = 1,
        select_grad: bool = False,
        steps_per_epoch: int = None,
        arm_distribution: str = "horizontal",
        num_arms: int = 8,
        num_arms_to_select: int = 2,
        ucb_alpha: float = 0.5,
        arm_selection_method: str = "ucb",
        thompson_distribution: str = "beta",
        thompson_alpha_prior: float = 1.0,
        thompson_beta_prior: float = 1.0,
        thompson_gaussian_prior_mean: float = 0.0,
        thompson_gaussian_prior_std: float = 1.0,
        thompson_gaussian_noise_std: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        PacaLayer.__init__(self, base_layer, arm_distribution=arm_distribution, num_arms=num_arms,
                         num_arms_to_select=num_arms_to_select, ucb_alpha=ucb_alpha,
                         arm_selection_method=arm_selection_method,
                         thompson_distribution=thompson_distribution,
                         thompson_alpha_prior=thompson_alpha_prior,
                         thompson_beta_prior=thompson_beta_prior,
                         thompson_gaussian_prior_mean=thompson_gaussian_prior_mean,
                         thompson_gaussian_prior_std=thompson_gaussian_prior_std,
                         thompson_gaussian_noise_std=thompson_gaussian_noise_std,
                         **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            paca_alpha=paca_alpha,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.iterations = 0
        self.select_grad = select_grad
        self.done_selection = False
        self.steps_per_epoch = steps_per_epoch

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.paca_w.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights.data[:, self.paca_indices[active_adapter]] = (
                        self.scaling[active_adapter] * self.paca_w[active_adapter].to(orig_weights.dtype)
                    )
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data[:, self.paca_indices[active_adapter]] = (
                        self.scaling[active_adapter] * self.paca_w[active_adapter].to(base_layer.weight.dtype)
                    )
                self.merged_adapters.append(active_adapter)
                
    def paca_update(self, adapter_name):
        with torch.no_grad():
            self.weight.data[:, self.paca_indices[adapter_name]] = (
                self.scaling[adapter_name] * self.paca_w[adapter_name].to(self.weight.dtype)
            )
                
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # Reshape input for VMamba
        original_shape = x.shape
        if x.dim() == 4:
            if self.in_features == original_shape[1]:
                x = x.permute(0, 2, 3, 1).contiguous()
            batch_size, height, width, channels = x.shape
            x = x.view(batch_size * height * width, channels)
        elif x.dim() == 3:
            batch_size, seq_len, channels = x.shape
            x = x.view(batch_size * seq_len, channels)

        # Check if we need to reselect indices (ONLY for horizontal distribution)
        if self.steps_per_epoch is not None and self.training and self.arm_distribution == "horizontal":
            for active_adapter in self.active_adapters:
                if active_adapter not in self.paca_w.keys():
                    continue
                if self.iterations % (self.steps_per_epoch * self.gradient_accumulation_steps) == 0 and self.iterations > 0:
                    self.reselect_ucb_indices(active_adapter)

        self.iterations += 1
        if adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.paca_w.keys():
                    continue
                if self.training and self.iterations % (2 * self.gradient_accumulation_steps):
                    self.paca_update(active_adapter)
                paca_w = self.paca_w[active_adapter] * self.scaling[active_adapter]
                result = pacalinear.apply(
                    x, self.weight, paca_w, self.bias, self.paca_indices[active_adapter], self.select_grad
                )

        # Reshape output back to original spatial dimensions
        if len(original_shape) == 4:
            if self.in_features == original_shape[1]:
                result = result.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
            else:
                result = result.view(batch_size, height, width, -1)
        elif len(original_shape) == 3:
            result = result.view(batch_size, seq_len, -1)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "paca." + rep

class pacalinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, paca_w=None, bias=None, paca_indices=None, select_grad=False):
        if input.dim() == 2 and bias is not None:
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output
        if not select_grad:
            ctx.save_for_backward(prune_activation(input, paca_indices), weight, bias)
        else:
            ctx.save_for_backward(input, weight, bias)
        ctx.select_grad = select_grad
        return ret

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        if not ctx.select_grad:
            pruned_input, weight, bias = ctx.saved_tensors
        else:
            pruned_input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_paca_w = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        
        if not ctx.select_grad:
            if ctx.needs_input_grad[2]:
                dim = grad_output.dim()
                if dim > 2:
                    grad_paca_w = grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
                        pruned_input.reshape(-1, pruned_input.shape[-1])
                    )
                else:
                    grad_paca_w = grad_output.t().matmul(pruned_input)
        else:
            dim = grad_output.dim()
            if dim > 2:
                grad_tensor = grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
                    pruned_input.reshape(-1, pruned_input.shape[-1])
                )
            else:
                grad_tensor = grad_output.t().matmul(pruned_input)
            
        if bias is not None and ctx.needs_input_grad[2]:
            if dim > 2:
                grad_bias = grad_output.sum([i for i in range(dim - 1)])
            else:
                grad_bias = grad_output.sum(0)
        return grad_input, None, grad_paca_w, grad_bias, None, None, None

def prune_weight(weight, indices=None):
    if indices is None:
        raise ValueError("Indices must be provided to prune activation dimensions.")
    pruned_weight = weight[:, indices]
    return pruned_weight

def prune_activation(activation, indices=None):
    if indices is None:
        raise ValueError("Indices must be provided to prune activation dimensions.")
    if activation.dim() == 2:
        return activation[:, indices]
    elif activation.dim() == 3:
        return activation[:, :, indices]
    elif activation.dim() == 4:
        B, C, H, W = activation.shape
        activation = activation.view(B, C, H * W).transpose(1, 2)
        return activation[:, :, indices]
    else:
        raise ValueError(f"Unsupported activation dim: {activation.dim()}")

def dispatch_default(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None
    bias = kwargs.pop("bias", False)
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target
    new_module = Linear(
        target,
        adapter_name,
        bias=bias,
        **kwargs,
    )
    return new_module
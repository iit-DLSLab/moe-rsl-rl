# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from rsl_rl.utils import resolve_callable

from .ac_moe_common import BaseMoENet, DiagonalGaussianMixture, MaskedActionNormal
from .ac_moe_explicit import ExplicitExpertMoENet, _ExplicitExpertOnnxModel
from .ac_moe_gated import GatedMoENet


def MoE_net(
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    gate_hidden_dims: list[int] | None = None,
    activation: str = "elu",
    num_experts: int = 4,
    top_k: int | None = -1,
    use_gate_loss: bool = False,
    use_load_balance_loss: bool = False,
    use_explicit_expert: bool = False,
    use_shared_layers: bool | str = False,
    expert_output_dims: list[int] | None = None,
) -> BaseMoENet:
    """Build either a learned-gate or explicitly routed MoE network."""
    # Hard routing: the expert index comes from the observation, no learned gate is built
    if use_explicit_expert:
        return ExplicitExpertMoENet(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            num_experts=num_experts,
            use_gate_loss=use_gate_loss,
            use_load_balance_loss=use_load_balance_loss,
            use_shared_layers=use_shared_layers,
            expert_output_dims=expert_output_dims,
        )

    if expert_output_dims is not None:
        raise ValueError("`expert_output_dims` is supported only with `use_explicit_expert=True`.")

    # Soft routing: a learned gate produces the mixture weights over experts
    return GatedMoENet(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        gate_hidden_dims=gate_hidden_dims,
        activation=activation,
        num_experts=num_experts,
        top_k=-1 if top_k is None else top_k,
        use_gate_loss=use_gate_loss,
        use_load_balance_loss=use_load_balance_loss,
        use_shared_layers=use_shared_layers,
    )


class MoEDistribution(Distribution):
    """Gaussian output distribution with one standard deviation vector per expert."""

    def __init__(
        self,
        output_dim: int,
        moe: BaseMoENet,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1.0e-3, 2.0),
        std_type: str = "scalar",
        learn_std: bool = True,
        use_gaussian_mixture: bool = False,
    ) -> None:
        """Initialize the MoE-aware action distribution."""
        super().__init__(output_dim)
        # Avoid registering the same expert network both as the model MLP and as a child of the distribution.
        object.__setattr__(self, "_moe", moe)
        self.std_type = std_type
        self.std_range = (max(float(std_range[0]), 1.0e-6), float(std_range[1]))
        # Precompute the log-space bounds so `log` std parameters can be clamped without repeated log() calls
        self.log_std_range = (
            float(torch.log(torch.tensor(self.std_range[0]))),
            float(torch.log(torch.tensor(self.std_range[1]))),
        )
        self.use_gaussian_mixture = use_gaussian_mixture

        # One standard deviation vector per expert, learned jointly with the policy
        shape = (moe.num_experts, output_dim)
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(shape), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(
                torch.log(init_std * torch.ones(shape)), requires_grad=learn_std
            )
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Populated by `update()` after every forward pass, once the router weights are known
        self._distribution: Normal | MaskedActionNormal | DiagonalGaussianMixture | None = None
        Normal.set_default_validate_args(False)

    @property
    def moe(self) -> BaseMoENet:
        """Return the MoE network associated with this distribution."""
        return self._moe  # type: ignore[attr-defined]

    def _expert_std(self, batch_size: int) -> torch.Tensor:
        if self.std_type == "scalar":
            expert_std = self.std_param.clamp(self.std_range[0], self.std_range[1])
        else:
            log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
            expert_std = torch.exp(log_std)
        return expert_std.unsqueeze(0).expand(batch_size, -1, -1)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the action distribution from the latest MoE forward pass."""
        expert_std = self._expert_std(mlp_output.shape[0])

        # Gaussian-mixture mode keeps every expert as a separate mixture component instead of
        # collapsing the outputs into a single mean, so the action distribution is multi-modal.
        if self.use_gaussian_mixture:
            component_means = self.moe._last_component_outputs.transpose(1, 2)
            component_action_masks = self.moe.expert_action_masks if self.moe.has_variable_expert_outputs else None
            self._distribution = DiagonalGaussianMixture(
                component_means,
                expert_std,
                self.moe._last_gate_weights.squeeze(1),
                component_action_masks=component_action_masks,
            )
            return

        action_mask: torch.Tensor | None = None
        if self.moe.use_explicit_expert:
            # Hard routing already collapsed the mean to the selected expert; pick the matching std
            selector = self.moe._last_gate_weights.squeeze(1).argmax(dim=-1)
            batch_idx = torch.arange(mlp_output.shape[0], device=mlp_output.device)
            std = expert_std[batch_idx, selector]
            if self.moe.has_variable_expert_outputs:
                action_mask = self.moe.expert_action_masks.index_select(0, selector).to(dtype=mlp_output.dtype)
        else:
            # Soft routing: blend the per-expert std by the same gate weights used for the mean
            weights = self.moe._last_gate_weights.squeeze(1)
            std = (weights.unsqueeze(-1) * expert_std).sum(dim=1)

        # Fall back to a plain Normal when every expert controls the full action vector
        if action_mask is None:
            self._distribution = Normal(mlp_output, std)
        else:
            self._distribution = MaskedActionNormal(mlp_output, std, action_mask)

    def sample(self) -> torch.Tensor:
        """Sample actions from the current distribution."""
        return self._distribution.sample()  # type: ignore[union-attr]

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the deterministic MoE output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return the identity used by deterministic policy exports."""
        return nn.Identity()

    @property
    def input_dim(self) -> int:
        """Return the number of action means emitted by the MoE."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the current action mean."""
        return self._distribution.mean  # type: ignore[union-attr]

    @property
    def std(self) -> torch.Tensor:
        """Return the current action standard deviation."""
        return self._distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        """Return action entropy summed over active action dimensions."""
        entropy = self._distribution.entropy()  # type: ignore[union-attr]
        return entropy.sum(dim=-1) if entropy.dim() > 1 else entropy

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return mean and standard deviation for rollout storage."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return action log probability summed over active dimensions."""
        log_prob = self._distribution.log_prob(outputs)  # type: ignore[union-attr]
        return log_prob.sum(dim=-1) if log_prob.dim() > 1 else log_prob

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute diagonal-Gaussian KL while ignoring inactive action dimensions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        # A zero std marks a masked-out action dimension for a given sample; exclude it from the KL
        active_dims = (old_std > 0.0) & (new_std > 0.0)
        old_std = old_std.clamp_min(1.0e-6)
        new_std = new_std.clamp_min(1.0e-6)
        std_ratio = new_std / old_std
        log_std_ratio = (
            torch.log(std_ratio) if self.moe.has_variable_expert_outputs else torch.log(std_ratio + 1.0e-5)
        )
        # Standard closed-form KL between two diagonal Gaussians, summed over active dimensions only
        kl = (
            log_std_ratio
            + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
            - 0.5
        )
        return (kl * active_dims).sum(dim=-1)


class MoEModel(MLPModel):
    """RSL-RL v5 MLP-model interface backed by a Mixture-of-Experts network."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        gate_hidden_dims: list[int] | None = None,
        num_experts: int = 4,
        top_k: int | None = -1,
        use_gate_loss: bool = False,
        use_load_balance_loss: bool = False,
        use_explicit_expert: bool = False,
        use_shared_layers: bool | str = False,
        expert_output_dims: list[int] | None = None,
        use_gaussian_mixture: bool = False,
    ) -> None:
        """Initialize an actor or critic MoE model."""
        # Skip MLPModel.__init__: it builds a plain MLP, whereas here `self.mlp` is a MoE network
        nn.Module.__init__(self)

        # Resolve the observation groups feeding this model (actor or critic) and their flattened size
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()

        # Build the expert network, keeping the RSL-RL `self.mlp` attribute name so export utilities work
        self.mlp = MoE_net(
            obs_dim=self.obs_dim,
            act_dim=output_dim,
            hidden_dims=hidden_dims,
            gate_hidden_dims=gate_hidden_dims,
            activation=activation,
            num_experts=num_experts,
            top_k=top_k,
            use_gate_loss=use_gate_loss,
            use_load_balance_loss=use_load_balance_loss,
            use_explicit_expert=use_explicit_expert,
            use_shared_layers=use_shared_layers,
            expert_output_dims=expert_output_dims,
        )

        # Record which auxiliary PPO losses and KL/log-prob strategies apply to this model
        self.use_gate_loss = use_gate_loss
        self.use_load_balance_loss = use_load_balance_loss and not use_explicit_expert
        self.use_gaussian_mixture = use_gaussian_mixture
        self.use_variable_expert_outputs = self.mlp.has_variable_expert_outputs
        self.use_masked_action_kl = self.use_variable_expert_outputs and not use_gaussian_mixture
        self.use_log_prob_kl = use_gaussian_mixture

        # A critic has no distribution config; only a stochastic actor builds a MoE-aware distribution
        if distribution_cfg is None:
            if use_gaussian_mixture:
                raise ValueError("`use_gaussian_mixture=True` is valid only for a stochastic actor model.")
            self.distribution = None
        else:
            distribution_cfg = distribution_cfg.copy()
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            if dist_class is not GaussianDistribution:
                raise ValueError("MoE actors currently support only `GaussianDistribution`.")
            self.distribution = MoEDistribution(
                output_dim,
                self.mlp,
                use_gaussian_mixture=use_gaussian_mixture,
                **distribution_cfg,
            )

    def gate_entropy(self) -> torch.Tensor:
        """Return the mean entropy of the most recent routing weights."""
        weights = self.mlp._last_gate_weights
        return -(weights * torch.log(weights + 1.0e-8)).sum(dim=-1).mean()

    def load_balance_loss(self) -> torch.Tensor:
        """Return the load-balancing loss from the MoE router."""
        return self.mlp.load_balance_loss()

    def _export_with_cleared_router_cache(self, build_fn) -> nn.Module:
        """Run `build_fn` without deep-copying tensors from the latest autograd graph."""
        cache_names = ("_last_gate_weights", "_last_unmasked_gate_weights", "_last_component_outputs")
        cached_tensors = {name: getattr(self.mlp, name) for name in cache_names}
        try:
            for name, value in cached_tensors.items():
                setattr(self.mlp, name, value.new_empty(0))
            return build_fn()
        finally:
            for name, value in cached_tensors.items():
                setattr(self.mlp, name, value)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible deterministic MoE policy."""
        parent_as_jit = super().as_jit
        return self._export_with_cleared_router_cache(parent_as_jit)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-compatible deterministic MoE policy."""
        if isinstance(self.mlp, ExplicitExpertMoENet):
            # Explicit routing only ever needs one expert per sample: export a batch-size-1
            # graph that conditionally dispatches to the selected expert instead of the
            # dense all-experts path used for training and for `as_jit`.
            return self._export_with_cleared_router_cache(lambda: torch.jit.script(_ExplicitExpertOnnxModel(self)))
        parent_as_onnx = super().as_onnx
        return self._export_with_cleared_router_cache(lambda: parent_as_onnx(verbose))

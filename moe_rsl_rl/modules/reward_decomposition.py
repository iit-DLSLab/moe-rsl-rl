# Copyright (c) 2026, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class RewardDecompositionCritic(nn.Module):
    """Independent critic networks, one per reward group.

    Each critic is a regular RSL-RL v5 critic model (``MLPModel`` or ``MoEModel``) with a single output that estimates
    the value of its own reward group. The forward pass concatenates the critics' outputs along the last dimension, in
    the order of ``critics``, so the PPO rollout and update code can treat the values as a ``(..., num_critics)``
    tensor.
    """

    is_recurrent: bool = False

    def __init__(self, critics: dict[str, nn.Module]) -> None:
        """Wrap one critic per reward group, keyed by the group name."""
        super().__init__()
        if not critics:
            raise ValueError("RewardDecompositionCritic requires at least one critic.")
        if any(critic.is_recurrent for critic in critics.values()):
            raise ValueError("The reward decomposition currently supports feed-forward critics only.")
        self.critics = nn.ModuleDict(critics)
        self.group_names = list(critics)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return the stacked values of all critics with shape ``(..., num_critics)``."""
        return torch.cat([critic(obs, masks=masks) for critic in self.critics.values()], dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the critics (no-op for feed-forward critics)."""
        for critic in self.critics.values():
            critic.reset(dones)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for feed-forward critics)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state (no-op for feed-forward critics)."""
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the observation normalizer of every critic."""
        for critic in self.critics.values():
            critic.update_normalization(obs)

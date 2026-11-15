# Copyright (c) 2026, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class RewardDecompositionRolloutStorage(RolloutStorage):
    """Rollout storage with one reward, value, and return channel per critic.

    ``group_rewards``, ``values``, and ``returns`` have shape ``(num_steps, num_envs, num_critics)``, while
    ``advantages`` keeps the upstream ``(num_steps, num_envs, 1)`` shape because the actor is trained on a single
    combined advantage. ``rewards`` keeps the upstream shape and stores the sum over reward groups.
    """

    def __init__(
        self,
        num_critics: int,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate the RL rollout buffers with ``num_critics`` value channels."""
        super().__init__("rl", num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.num_critics = num_critics
        shape = (num_transitions_per_env, num_envs, num_critics)
        self.group_rewards = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        self.returns = torch.zeros(shape, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store a transition whose ``rewards`` and ``values`` have shape ``(num_envs, num_critics)``."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        self.group_rewards[self.step].copy_(transition.rewards)  # type: ignore[arg-type]
        # The upstream buffer stores one reward per environment: keep the total there
        transition.rewards = transition.rewards.sum(dim=-1)  # type: ignore[union-attr]
        super().add_transition(transition)

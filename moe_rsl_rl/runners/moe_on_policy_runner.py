# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from moe_rsl_rl.algorithms import PPO as MoEPPO


class MoEOnPolicyRunner(OnPolicyRunner):
    """RSL-RL on-policy runner selecting the MoE-aware PPO implementation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Use the local PPO for MoE configs, then delegate all runner behavior to RSL-RL."""
        # Select the MoE-aware algorithm while keeping the standard RSL-RL configuration shape
        if "moe_cfg" in train_cfg:
            train_cfg["algorithm"]["class_name"] = MoEPPO

        # Delegate construction, learning, logging, checkpoints, and exports to the upstream runner
        super().__init__(env, train_cfg, log_dir, device)

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import torch
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

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file.

        The explicit-expert MoE export (`_ExplicitExpertOnnxModel`) is a `torch.jit.script`
        module: it dispatches to a single expert with a data-dependent `if i == selector`
        branch so that only the selected expert's forward pass runs, relying on TorchScript's
        control-flow tracing to emit ONNX `If` nodes. Starting with PyTorch 2.9, `torch.onnx.export`
        defaults to the `torch.export`-based exporter, which cannot accept an already-scripted
        module (`ValueError: Exporting a ScriptModule is not supported`). Force the legacy
        TorchScript-based exporter, which this scripted control-flow pattern was written for.
        """
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
            dynamo=False,
        )

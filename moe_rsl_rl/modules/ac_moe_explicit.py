from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn as nn

from .ac_moe_common import BaseMoENet

if TYPE_CHECKING:
    from .ac_moe import MoEModel


class ExplicitExpertMoENet(BaseMoENet):
    """MoE network with hard expert selection encoded in the last observation entry."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims,
        activation="elu",
        num_experts: int = 4,
        use_gate_loss: bool = False,
        use_load_balance_loss: bool = False,
        use_shared_layers="None",
        expert_output_dims: list[int] | None = None,
    ):
        # Build the same expert topology while reserving the final observation as selector
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            num_experts=num_experts,
            use_explicit_expert=True,
            use_shared_layers=use_shared_layers,
            expert_output_dims=expert_output_dims,
        )
        # Preserve the gate-entropy option for compatibility with the previous implementation
        self.use_gate_loss = use_gate_loss
        # Explicit routing does not optimize gate balancing.
        self.use_load_balance_loss = False
        self.top_k = -1
        self.is_sparse = False

    def _gate_explicit(self, x: torch.Tensor) -> torch.Tensor:
        # Convert the last observation entry into one-hot routing weights
        selector_vals = x[:, -1].round().long().clamp(0, self.num_experts - 1)
        weights = torch.zeros(x.shape[0], self.num_experts, device=x.device)
        weights.scatter_(1, selector_vals.unsqueeze(1), 1.0)
        return weights.unsqueeze(1)

    def forward(self, x: torch.Tensor, return_gate: bool = False) -> torch.Tensor:
        # Evaluate all expert components and select exactly one output per environment
        expert_out, _ = self._compute_experts(x)
        weights = self._gate_explicit(x)

        # Cache routing and component outputs for the MoE-aware action distribution
        self._last_gate_weights = weights
        component_out = self._component_outputs(expert_out)
        self._last_component_outputs = component_out
        return self._combine_direct(component_out, weights)

    def load_balance_loss(self) -> torch.Tensor:
        # Explicit routing has no trainable router to balance
        return torch.zeros((), device=self._last_gate_weights.device)


class _ExplicitExpertHeadDispatch(nn.Module):
    """Evaluate only the expert head selected by `idx`.

    `torch.jit.script` unrolls the `ModuleList` iteration at compile time, so each expert
    is checked with a literal, statically-resolved branch; the comparison against the
    runtime `idx` then compiles to a chain of ONNX `If` nodes, and only the branch that
    matches actually runs its expert's forward pass.
    """

    def __init__(self, heads: nn.ModuleList, output_dims: list[int], out_width: int) -> None:
        super().__init__()
        self.heads = heads
        self.output_dims = output_dims
        self.out_width = out_width

    def forward(self, head_input: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        selector = int(idx.item())
        out = head_input.new_zeros(head_input.shape[0], self.out_width)
        for i, head in enumerate(self.heads):
            if i == selector:
                head_out = head(head_input)
                dim = self.output_dims[i]
                if dim == self.out_width:
                    out = head_out
                else:
                    padded = head_input.new_zeros(head_input.shape[0], self.out_width)
                    padded[:, :dim] = head_out
                    out = padded
        return out


class _ExplicitExpertOnnxModel(nn.Module):
    """Batch-size-1 ONNX export wrapper for explicit-routing MoE policies.

    The dense training-time forward evaluates every expert because a batch of parallel
    environments can each route to a different one. A deployed policy, however, only ever
    sees one observation per step, so at export time the selected expert can be dispatched
    conditionally instead of paying for every expert's forward pass.
    """

    is_recurrent: bool = False

    def __init__(self, model: "MoEModel") -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp: ExplicitExpertMoENet = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.input_size = model.obs_dim
        self.input_names = ["obs"]
        self.output_names = ["actions"]

        mlp = self.mlp
        if mlp.use_shared_backbone_and_head:
            # Every expert head feeds the shared output head at a common width.
            out_width = mlp.shared_head.in_features
            output_dims = [out_width for _ in range(mlp.num_experts)]
        else:
            out_width = mlp.act_dim
            output_dims = mlp.expert_output_dims
        self.dispatch = _ExplicitExpertHeadDispatch(mlp.experts, output_dims, out_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        obs_input = self.mlp._prepare_observation_input(x)
        selector_vals = x[:, -1].round().long().clamp(0, self.mlp.num_experts - 1)

        if self.mlp.use_shared_backbone or self.mlp.use_shared_backbone_and_head:
            head_input = self.mlp.shared_backbone(obs_input)
        else:
            head_input = obs_input

        component_out = self.dispatch(head_input, selector_vals)
        if self.mlp.use_shared_backbone_and_head:
            component_out = self.mlp.shared_head(component_out)

        if self.mlp.has_variable_expert_outputs:
            mask = self.mlp.expert_action_masks.index_select(0, selector_vals).to(dtype=component_out.dtype)
            component_out = component_out * mask

        return self.deterministic_output(component_out)

    @torch.jit.export
    def get_dummy_inputs(self) -> Tuple[torch.Tensor]:
        """Return a batch-size-1 dummy observation for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

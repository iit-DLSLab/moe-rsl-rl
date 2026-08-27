# How to Use moe-rsl-rl

This guide explains what is inside this repository and how to wire it into an IsaacLab/RSL-RL training setup. The short version is:

1. Install this package in the same Python environment you use for IsaacLab.
2. Replace the standard RSL-RL runner with the MoE runner from this repository.
3. Configure the standard RSL-RL v5 `actor` and `critic` models.
4. Add a top-level `moe_cfg` block to your training config.
5. Select the MoE side with `moe_cfg.who` and choose learned or explicit routing.

The key idea is that `moe-rsl-rl` does not define an IsaacLab task. It extends the RSL-RL PPO training stack with Mixture-of-Experts actor/critic modules and a local PPO fork that knows how to handle the extra MoE losses and distributions.


## Minimal Config Skeleton to add in your isaaclab!

See in the folder example!


## Repository Layout

```text
moe_rsl_rl/
  algorithms/
    ppo.py                    # Local PPO fork with MoE auxiliary losses and MoE-aware KL handling.
  modules/
    ac_moe.py                 # RSL-RL v5 MoEModel, distribution, and network factory.
    ac_moe_common.py          # Shared expert helpers, masking, and Gaussian-mixture distributions.
    ac_moe_gated.py           # Learned-gate MoE network, with optional top-k sparse routing.
    ac_moe_explicit.py        # Hard-routed MoE network using the last observation entry as expert id.
  runners/
    moe_on_policy_runner.py   # Upstream v5.4.2 runner with local PPO selection for MoE configs.
example/
  moe_cfg.py                  # Example MoE configuration.
  rsl_rl_ppo_cfg.py           # Example RSL-RL v5 actor/critic and PPO configuration.
README.md                    # Project overview.
pyproject.toml               # Package metadata and dependencies.
```

This repository is a library layer. It does not currently ship a complete IsaacLab task, environment, or standalone `train.py`. Your IsaacLab project still owns the task definition, observation tensors, rewards, commands, terrain, logging setup, and train/play scripts.


## How the Pieces Fit Together

At runtime the flow is:

```text
IsaacLab VecEnv
  -> MoE on-policy runner
    -> local PPO
      -> separate actor and critic models (RSL-RL v5 API)
      -> learned-gate MoE, explicit-expert MoE, or regular MLP per side
      -> standard PPO loss
      -> optional gate-entropy and load-balancing auxiliary losses
```

The runner expects a training config with these top-level sections:

```python
train_cfg = {
    "actor": {...},
    "critic": {...},
    "algorithm": {...},
    "obs_groups": {...},
    "num_steps_per_env": ...,
    "save_interval": ...,
    "moe_cfg": {...},
    ...
}
```

The important parts are:

- `actor` and `critic` use the standard RSL-RL v5 model configuration shape.
- `algorithm["class_name"] == "PPO"` selects this package's RSL-RL 5.4.2-compatible PPO in the MoE runner.
- `moe_cfg["who"]` makes the actor, critic, or both use `MoEModel`.

## Choose a Workflow

There are two main ways to use the package.

### Option 1: Learned Gated MoE

Use this when you want the network to learn how much each expert should contribute from the observation.

```python
"moe_cfg": {
    "who": "actor+critic",
    "num_experts": 4,
    "gate_hidden_dims": [128],
    "top_k": None,
    "use_gate_loss": False,
    "use_load_balance_loss": False,
    "use_explicit_expert": False,
    "use_shared_layers": False,
}
```

With `use_explicit_expert=False`, the model builds a `GatedMoENet`. The gate reads the same observation features as the experts and outputs expert weights.

- `top_k=None`, `top_k=-1`, or `top_k >= num_experts` uses dense gating.
- `0 < top_k < num_experts` uses sparse top-k gating.
- `gate_hidden_dims=[]` or `None` makes the gate a single linear layer.
- `use_load_balance_loss=True` adds a small load-balancing auxiliary term in PPO.
- `use_gate_loss=True` adds a small gate-entropy term in PPO.

This is usually the easiest path to debug first because it does not require adding an explicit expert-id feature to the observation.

### Option 2: Explicit Expert Selection

Use this when your environment or curriculum already knows which expert should be active.

```python
"moe_cfg": {
    "who": "actor+critic",
    "num_experts": 4,
    "gate_hidden_dims": [128],
    "top_k": None,
    "use_gate_loss": False,
    "use_load_balance_loss": False,
    "use_explicit_expert": True,
    "use_shared_layers": "backbone",
    "expert_output_dims": [12, 9, 9, 6],
}
```

With `use_explicit_expert=True`, the model builds an `ExplicitExpertMoENet`. The last scalar in the MoE input is rounded, clamped to `[0, num_experts - 1]`, and used as the hard expert id. That selector is removed before the expert networks process the observation.

If `who` includes `actor`, the actor observation must end with the expert id. If `who` includes `critic`, the critic observation must also end with the expert id. If the selector is missing, the model will accidentally treat the last real observation feature as the expert id.

`expert_output_dims` is optional and only applies to the actor. It lets each explicit actor expert control the first `N` action dimensions and masks the remaining dimensions to zero. The list must contain one value per expert, and every value must be between `1` and `env.num_actions`.

## Use the MoE Runner

In your IsaacLab training script, use the runner in `moe_rsl_rl/runners/moe_on_policy_runner.py` instead of the standard RSL-RL on-policy runner.

```python
from moe_rsl_rl.runners.moe_on_policy_runner import MoEOnPolicyRunner

runner = MoEOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
```

The runner is kept aligned with RSL-RL 5.4.2. Its MoE-specific behavior is deliberately limited to:

- selecting this repository's local `PPO` when a top-level `moe_cfg` is present
- exposing the class as `MoEOnPolicyRunner`

The local PPO constructs `MoEModel` on the sides selected by `who` and retains the standard v5 runner lifecycle,
checkpoint format, logging, multi-GPU support, and JIT/ONNX entry points.

Checkpoints written by this version use the RSL-RL v5 `actor_state_dict`/`critic_state_dict` layout. Loading an older
v3 `model_state_dict` is supported for model weights and iteration count; its PPO optimizer is reinitialized because
the v5 parameter layout is different.

## Configure Mixture of Experts

The MoE config is the most important part:

```python
"moe_cfg": {
    "who": "actor+critic",
    "num_experts": 4,
    "gate_hidden_dims": [128],
    "top_k": None,
    "use_gate_loss": False,
    "use_load_balance_loss": False,
    "use_explicit_expert": True,
    "use_shared_layers": "backbone",
    "expert_output_dims": [12, 9, 9, 6],
    # Optional:
    # "use_gaussian_mixture": False,
}
```

Treat this config as a contract between your IsaacLab observation/action layout and the MoE code:

- `who` decides where the MoE is used. Supported values should contain `"actor"`, `"critic"`, or both, for example `"actor"`, `"critic"`, or `"actor+critic"`.
- `num_experts` is the number of experts in every MoE network.
- `gate_hidden_dims` controls the learned gate MLP for `GatedMoENet`.
- `top_k` controls sparse learned gating when `use_explicit_expert=False`.
- `use_gate_loss` enables a small gate-entropy auxiliary term.
- `use_load_balance_loss` enables a small expert load-balancing auxiliary term for learned gating.
- `use_explicit_expert` switches from learned routing to hard routing from the last observation entry.
- `use_shared_layers` can be `False`, `"backbone"`, or `"backbone+head"`.
- `expert_output_dims` is only valid for explicit actor MoE.
- `use_gaussian_mixture=True` makes the actor distribution a mixture of diagonal Gaussians instead of a weighted mean with a single diagonal Gaussian.

## Choose Where MoE Is Applied

Set `who` based on which part of the actor-critic should use experts.

```text
actor          # MoE actor, regular MLP critic
critic         # regular MLP actor, MoE critic
actor+critic   # MoE actor and MoE critic
```

Use `"actor"` when you mainly want specialized policies with a conventional value function. Use `"critic"` when value estimation needs specialization but the action policy should remain a regular MLP. Use `"actor+critic"` when both sides should specialize.

These are the only accepted values; synonyms such as `"both"` are rejected.

## Set the Expert Routing Input

For learned gating, no special observation feature is required. The gate receives the same flattened observation input as the experts.

For explicit expert selection, append one scalar expert id to each MoE input:

```text
[regular observation features..., expert_id]
```

The expert id should be:

- `0` for expert 0
- `1` for expert 1
- up to `num_experts - 1`

The code rounds the value and clamps it into range. Keep it as a clean scalar in the observation anyway; noisy or normalized expert ids make debugging unnecessarily confusing.

If only the actor is MoE, append the selector only to the actor/policy observation. If only the critic is MoE, append it only to the critic observation. If both are MoE, append it to both.

## Supported Model Settings

`MoEModel` supports the RSL-RL v5 MLP settings `hidden_dims`, `activation`, and `obs_normalization`. A stochastic MoE
actor currently requires `GaussianDistribution`; its `init_std`, `std_type`, `std_range`, and `learn_std` options are
supported. The v5 heteroscedastic and beta distributions are not MoE-enabled. The critic remains deterministic.

## Match Config to Tensor Dimensions

The MoE config must agree with the actual tensor sizes.

For example, if your environment has 12 actions and you use:

```python
"expert_output_dims": [12, 9, 9, 6]
```

then:

- `num_experts` must be `4`
- `env.num_actions` must be at least `12`
- expert 0 can output all 12 action dimensions
- expert 1 and expert 2 output the first 9 action dimensions
- expert 3 outputs the first 6 action dimensions
- inactive action dimensions are masked to zero for explicit variable-output experts

For explicit routing, remember that the selector increases the corresponding observation dimension by one. The model removes that final scalar internally before passing features through the expert networks.

For shared expert layouts:

- `use_shared_layers=False` gives each expert its own MLP.
- `use_shared_layers="backbone"` shares the early layers, then gives each expert its own head.
- `use_shared_layers="backbone+head"` shares the early layers and final action/value head.

When using `"backbone"` or `"backbone+head"`, use at least two hidden dimensions so the shared backbone and expert head have valid layer sizes.

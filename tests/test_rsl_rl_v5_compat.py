# Copyright (c) 2026, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner

from moe_rsl_rl.algorithms import PPO
from moe_rsl_rl.modules import MoEModel
from moe_rsl_rl.runners import MoEOnPolicyRunner


NUM_ENVS = 6
OBS_DIM = 7
NUM_ACTIONS = 4


class DummyEnv(VecEnv):
    """Small CPU-only environment for runner and algorithm integration tests."""

    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        observations = torch.randn(self.num_envs, OBS_DIM)
        observations[:, -1] = torch.arange(self.num_envs) % 3
        return TensorDict({"policy": observations}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        del actions
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        extras = {"time_outs": torch.zeros(self.num_envs)}
        return self.get_observations(), torch.randn(self.num_envs), dones, extras


def make_cfg(
    who: str = "actor+critic",
    *,
    explicit: bool = False,
    gaussian_mixture: bool = False,
    expert_output_dims: list[int] | None = None,
) -> dict:
    """Return a minimal RSL-RL 5.4.2 MoE configuration."""
    return {
        "num_steps_per_env": 4,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "schedule": "adaptive",
        },
        "moe_cfg": {
            "who": who,
            "num_experts": 3,
            "gate_hidden_dims": [8],
            "top_k": 2,
            "use_gate_loss": True,
            "use_load_balance_loss": True,
            "use_explicit_expert": explicit,
            "use_shared_layers": False,
            "expert_output_dims": expert_output_dims,
            "use_gaussian_mixture": gaussian_mixture,
        },
    }


def build_algorithm(env: DummyEnv, cfg: dict) -> PPO:
    """Construct a local PPO without mutating the caller's config."""
    copied_cfg = copy.deepcopy(cfg)
    copied_cfg["multi_gpu"] = None
    return PPO.construct_algorithm(env.get_observations(), env, copied_cfg, "cpu")


def run_update(algorithm: PPO, env: DummyEnv) -> dict[str, float]:
    """Collect one rollout and perform one PPO update."""
    for _ in range(algorithm.storage.num_transitions_per_env):
        observations = env.get_observations()
        algorithm.act(observations)
        algorithm.process_env_step(
            observations,
            torch.randn(env.num_envs),
            torch.zeros(env.num_envs),
            {},
        )
    algorithm.compute_returns(env.get_observations())
    return algorithm.update()


class TestRslRlV5Compatibility(unittest.TestCase):
    """Regression coverage for the RSL-RL 5.4.2 integration."""

    def test_local_ppo_extends_upstream_ppo(self) -> None:
        self.assertTrue(issubclass(PPO, RslRlPPO))
        self.assertTrue(issubclass(MoEOnPolicyRunner, OnPolicyRunner))

    def test_moe_is_constructed_only_on_selected_sides(self) -> None:
        env = DummyEnv()
        expected_types = {
            "actor": (MoEModel, MLPModel),
            "critic": (MLPModel, MoEModel),
            "actor+critic": (MoEModel, MoEModel),
        }
        for who, (actor_type, critic_type) in expected_types.items():
            with self.subTest(who=who):
                algorithm = build_algorithm(env, make_cfg(who))
                self.assertIsInstance(algorithm.actor, actor_type)
                self.assertIsInstance(algorithm.critic, critic_type)

    def test_all_previous_moe_distribution_modes_update(self) -> None:
        env = DummyEnv()
        modes = [
            make_cfg("actor", explicit=False),
            make_cfg("critic", explicit=False),
            make_cfg("actor+critic", explicit=False),
            make_cfg("actor", explicit=True, expert_output_dims=[4, 3, 2]),
            make_cfg("actor", gaussian_mixture=True),
            make_cfg("actor", explicit=True, gaussian_mixture=True, expert_output_dims=[4, 3, 2]),
        ]
        for cfg in modes:
            with self.subTest(moe_cfg=cfg["moe_cfg"]):
                losses = run_update(build_algorithm(env, cfg), env)
                self.assertEqual(set(losses), {"value", "surrogate", "entropy"})
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))

    def test_sparse_gate_and_explicit_action_masks_are_preserved(self) -> None:
        env = DummyEnv()

        sparse_algorithm = build_algorithm(env, make_cfg("actor"))
        sparse_algorithm.actor(env.get_observations(), stochastic_output=True)
        weights = sparse_algorithm.actor.mlp._last_gate_weights.squeeze(1)
        self.assertTrue(torch.all((weights > 0).sum(dim=-1) == 2))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(NUM_ENVS)))

        explicit_algorithm = build_algorithm(
            env,
            make_cfg("actor", explicit=True, expert_output_dims=[4, 3, 2]),
        )
        observations = env.get_observations()
        actions = explicit_algorithm.actor(observations, stochastic_output=True)
        expert_ids = observations["policy"][:, -1].long()
        for row, expert_id in zip(actions, expert_ids):
            active_dim = [4, 3, 2][expert_id]
            self.assertTrue(torch.equal(row[active_dim:], torch.zeros_like(row[active_dim:])))

    def test_runner_learn_checkpoint_and_policy_exports(self) -> None:
        env = DummyEnv()
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = MoEOnPolicyRunner(env, make_cfg("actor+critic"), log_dir=temp_dir, device="cpu")
            runner.learn(num_learning_iterations=1)
            self.assertIsInstance(runner.alg, PPO)

            checkpoint = Path(temp_dir) / "roundtrip.pt"
            runner.save(str(checkpoint), infos={"roundtrip": True})
            actor_state = {name: value.detach().clone() for name, value in runner.alg.actor.state_dict().items()}
            with torch.no_grad():
                next(runner.alg.actor.parameters()).add_(1.0)
            infos = runner.load(str(checkpoint))
            self.assertEqual(infos, {"roundtrip": True})
            self.assertTrue(
                all(torch.equal(actor_state[name], value) for name, value in runner.alg.actor.state_dict().items())
            )

            runner.export_policy_to_jit(temp_dir)
            self.assertTrue((Path(temp_dir) / "policy.pt").is_file())
            runner.export_policy_to_onnx(temp_dir)
            self.assertTrue((Path(temp_dir) / "policy.onnx").is_file())

    def test_v3_monolithic_checkpoint_weights_are_migrated(self) -> None:
        algorithm = build_algorithm(DummyEnv(), make_cfg("actor+critic"))
        expected_actor = {name: value.detach().clone() for name, value in algorithm.actor.state_dict().items()}
        expected_critic = {name: value.detach().clone() for name, value in algorithm.critic.state_dict().items()}

        legacy_state = {}
        for name, value in expected_actor.items():
            if name.startswith("mlp."):
                legacy_state[f"actor.{name.removeprefix('mlp.')}"] = value
            elif name.startswith("obs_normalizer."):
                legacy_state[f"actor_obs_normalizer.{name.removeprefix('obs_normalizer.')}"] = value
            elif name == "distribution.std_param":
                legacy_state["std"] = value
            elif name == "distribution.log_std_param":
                legacy_state["log_std"] = value
        for name, value in expected_critic.items():
            if name.startswith("mlp."):
                legacy_state[f"critic.{name.removeprefix('mlp.')}"] = value
            elif name.startswith("obs_normalizer."):
                legacy_state[f"critic_obs_normalizer.{name.removeprefix('obs_normalizer.')}"] = value

        with torch.no_grad():
            for parameter in algorithm.actor.parameters():
                parameter.add_(1.0)
            for parameter in algorithm.critic.parameters():
                parameter.add_(1.0)

        with self.assertWarnsRegex(UserWarning, "RSL-RL v3 checkpoint"):
            load_iteration = algorithm.load({"model_state_dict": legacy_state}, None, strict=True)

        self.assertTrue(load_iteration)
        self.assertTrue(
            all(torch.equal(expected_actor[name], value) for name, value in algorithm.actor.state_dict().items())
        )
        self.assertTrue(
            all(torch.equal(expected_critic[name], value) for name, value in algorithm.critic.state_dict().items())
        )


if __name__ == "__main__":
    unittest.main()

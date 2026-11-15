# Copyright (c) 2026, Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from moe_rsl_rl.algorithms import PPO
from moe_rsl_rl.modules import MoEModel, RewardDecompositionCritic
from moe_rsl_rl.runners import MoEOnPolicyRunner
from moe_rsl_rl.storage import RewardDecompositionRolloutStorage

NUM_ENVS = 8
OBS_DIM = 7
FEET_OBS_DIM = 3
NUM_ACTIONS = 4
GROUPS = ("feet", "rest")
REWARD_GROUPS = {"feet": ["feet_*"], "rest": ["*"]}


class GroupedRewardEnv(VecEnv):
    """CPU-only environment publishing its reward terms: two 'feet_*' terms and one other term."""

    def __init__(self, feet_reward: float | None = None, rest_reward: float | None = None) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 20
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.feet_reward = feet_reward
        self.rest_reward = rest_reward

    def get_observations(self) -> TensorDict:
        policy = torch.randn(self.num_envs, OBS_DIM)
        policy[:, -1] = torch.arange(self.num_envs) % 3  # explicit-expert selector
        feet = torch.randn(self.num_envs, FEET_OBS_DIM)
        return TensorDict({"policy": policy, "feet": feet}, batch_size=[self.num_envs])

    def reward_terms(self) -> dict[str, torch.Tensor]:
        def term(value: float | None) -> torch.Tensor:
            return torch.randn(self.num_envs) if value is None else torch.full((self.num_envs,), value)

        half_feet = None if self.feet_reward is None else self.feet_reward / 2
        return {
            "feet_air_time": term(half_feet),
            "lin_vel_tracking": term(self.rest_reward),
            "feet_slip": term(half_feet),
        }

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        del actions
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        self.episode_length_buf[time_outs] = 0
        terms = self.reward_terms()
        extras = {"time_outs": time_outs.float(), "reward_terms": terms}
        return self.get_observations(), sum(terms.values()), time_outs.float(), extras


def make_cfg(
    weights: dict[str, float] | None = None,
    *,
    normalize: bool = False,
    moe_cfg: dict | None = None,
    obs_groups: dict | None = None,
) -> dict:
    """Return a minimal RSL-RL 5.5.1 configuration with one critic per reward group."""
    cfg = {
        "num_steps_per_env": 6,
        "save_interval": 100,
        "obs_groups": obs_groups or {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16, 16], "activation": "elu", "obs_normalization": True},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 2, "num_mini_batches": 2, "schedule": "adaptive"},
        "moe_cfg": {
            "decomposition": "reward" if moe_cfg is None else "state+reward",
            "state_decomposition_cfg": moe_cfg,
            "reward_decomposition_cfg": {
                "groups": {
                    name: {"terms": terms, "weight": (weights or {}).get(name, 1.0)}
                    for name, terms in REWARD_GROUPS.items()
                },
                "normalize_advantages": normalize,
            },
        },
    }
    return cfg


def build_algorithm(env: VecEnv, cfg: dict) -> PPO:
    copied_cfg = copy.deepcopy(cfg)
    copied_cfg["multi_gpu"] = None
    return PPO.construct_algorithm(env.get_observations(), env, copied_cfg, "cpu")


def collect_rollout(algorithm: PPO, env: GroupedRewardEnv) -> TensorDict:
    """Fill the rollout storage, compute returns, and return the final observation."""
    obs = env.get_observations()
    for _ in range(algorithm.storage.num_transitions_per_env):
        actions = algorithm.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        algorithm.process_env_step(obs, rewards, dones, extras)
    algorithm.compute_returns(obs)
    return obs


def reference_gae(rewards, values, last_values, dones, gamma, lam) -> torch.Tensor:
    """Single-critic GAE (as in RSL-RL) on the given reward and value tensors of shape (T, N, 1)."""
    advantages = torch.zeros_like(rewards)
    advantage = 0
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + not_terminal * gamma * next_values - values[step]
        advantage = delta + not_terminal * gamma * lam * advantage
        advantages[step] = advantage
    return advantages


def normalize(tensor: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    return (tensor - tensor.mean(dim=dim)) / (tensor.std(dim=dim) + 1e-8)


class TestMultiCritic(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_constructs_one_independent_critic_per_group(self) -> None:
        algorithm = build_algorithm(GroupedRewardEnv(), make_cfg())
        self.assertIsInstance(algorithm.critic, RewardDecompositionCritic)
        self.assertIsInstance(algorithm.storage, RewardDecompositionRolloutStorage)
        self.assertEqual(list(algorithm.critic.critics), list(GROUPS))
        feet, rest = algorithm.critic.critics.values()
        self.assertIsInstance(feet, MLPModel)
        self.assertTrue(set(map(id, feet.parameters())).isdisjoint(map(id, rest.parameters())))
        self.assertEqual(algorithm.critic(GroupedRewardEnv().get_observations()).shape, (NUM_ENVS, len(GROUPS)))

    def test_unit_weights_match_single_critic_gae_of_the_total_reward(self) -> None:
        env = GroupedRewardEnv()
        algorithm = build_algorithm(env, make_cfg())
        last_obs = collect_rollout(algorithm, env)
        st = algorithm.storage
        with torch.no_grad():
            last_values = algorithm.critic(last_obs).sum(dim=-1, keepdim=True)
        # GAE is linear in rewards and values, so summing the groups (bootstrap included) must give the standard GAE
        expected = reference_gae(
            st.group_rewards.sum(dim=-1, keepdim=True),
            st.values.sum(dim=-1, keepdim=True),
            last_values,
            st.dones,
            algorithm.gamma,
            algorithm.lam,
        )
        torch.testing.assert_close(st.advantages, normalize(expected))
        torch.testing.assert_close(
            (st.returns - st.values).sum(dim=-1, keepdim=True), expected, rtol=1e-5, atol=1e-5
        )

    def test_normalized_weighted_group_advantages(self) -> None:
        env = GroupedRewardEnv()
        weights = {"feet": 2.0, "rest": 0.5}
        algorithm = build_algorithm(env, make_cfg(weights, normalize=True))
        collect_rollout(algorithm, env)
        st = algorithm.storage
        group_advantages = normalize(st.returns - st.values, dim=(0, 1))
        expected = normalize((group_advantages * torch.tensor([2.0, 0.5])).sum(dim=-1, keepdim=True))
        torch.testing.assert_close(st.advantages, expected)

    def test_each_critic_learns_the_value_of_its_own_group(self) -> None:
        env = GroupedRewardEnv(feet_reward=1.0, rest_reward=-1.0)
        cfg = make_cfg()
        cfg["algorithm"]["learning_rate"] = 1e-2
        algorithm = build_algorithm(env, cfg)
        for _ in range(30):
            collect_rollout(algorithm, env)
            losses = algorithm.update()
        self.assertTrue({"value", "value_feet", "value_rest"} <= set(losses))
        torch.testing.assert_close(losses["value"], losses["value_feet"] + losses["value_rest"])
        with torch.no_grad():
            values = algorithm.critic(env.get_observations()).mean(dim=0)
        self.assertGreater(values[0].item(), 1.0)
        self.assertLess(values[1].item(), -1.0)

    def test_per_group_observation_sets(self) -> None:
        obs_groups = {"actor": ["policy"], "critic": ["policy"], "critic_feet": ["feet"]}
        algorithm = build_algorithm(GroupedRewardEnv(), make_cfg(obs_groups=obs_groups))
        self.assertEqual(algorithm.critic.critics["feet"].obs_dim, FEET_OBS_DIM)
        self.assertEqual(algorithm.critic.critics["rest"].obs_dim, OBS_DIM)

    def test_moe_critics_with_pcgrad_update(self) -> None:
        env = GroupedRewardEnv()
        moe_cfg = {
            "who": "actor+critic",
            "num_experts": 3,
            "gate_hidden_dims": [8],
            "top_k": None,
            "use_explicit_expert": True,
            "use_shared_layers": "backbone",
            "use_pcgrad": True,
        }
        algorithm = build_algorithm(env, make_cfg(moe_cfg=moe_cfg))
        self.assertTrue(algorithm.use_pcgrad)
        critics = list(algorithm.critic.critics.values())
        self.assertTrue(all(isinstance(critic, MoEModel) for critic in critics))
        before = [copy.deepcopy(critic.state_dict()) for critic in critics]
        collect_rollout(algorithm, env)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        for critic, old in zip(critics, before):
            for prefix in ("mlp.shared_backbone", "mlp.experts.0", "mlp.experts.1", "mlp.experts.2"):
                self.assertTrue(any(
                    not torch.equal(parameter, old[name])
                    for name, parameter in critic.named_parameters() if name.startswith(prefix)
                ))

    def test_reward_terms_are_summed_into_their_first_matching_group(self) -> None:
        env = GroupedRewardEnv()
        algorithm = build_algorithm(env, make_cfg())
        obs = env.get_observations()
        algorithm.act(obs)
        terms = env.reward_terms()
        algorithm.process_env_step(obs, sum(terms.values()), torch.zeros(NUM_ENVS), {"reward_terms": terms})
        expected = torch.stack([terms["feet_air_time"] + terms["feet_slip"], terms["lin_vel_tracking"]], dim=-1)
        torch.testing.assert_close(algorithm.storage.group_rewards[0], expected)

    def test_decomposition_selects_the_sub_configs(self) -> None:
        env = GroupedRewardEnv()
        moe_cfg = {"who": "critic", "num_experts": 2, "gate_hidden_dims": [8], "top_k": None}
        cfg = make_cfg(moe_cfg=moe_cfg)
        cfg["moe_cfg"]["decomposition"] = "state"  # the reward decomposition is ignored
        algorithm = build_algorithm(env, cfg)
        self.assertIsInstance(algorithm.critic, MoEModel)
        self.assertIsInstance(algorithm.storage, RolloutStorage)
        self.assertFalse(algorithm.use_reward_decomposition)

        cfg["moe_cfg"]["decomposition"] = "reward"  # the state decomposition is ignored
        algorithm = build_algorithm(env, cfg)
        self.assertIsInstance(algorithm.critic, RewardDecompositionCritic)
        self.assertTrue(all(type(critic) is MLPModel for critic in algorithm.critic.critics.values()))

        cfg["moe_cfg"]["decomposition"] = "state+reward"  # one mixture of experts per reward group
        algorithm = build_algorithm(env, cfg)
        self.assertTrue(all(isinstance(critic, MoEModel) for critic in algorithm.critic.critics.values()))

    def test_misconfigured_reward_groups_raise(self) -> None:
        env = GroupedRewardEnv()
        misconfigurations = {
            "Unknown `moe_cfg` keys": {"who": "actor+critic"},
            "must be 'state', 'reward', or 'state\\+reward'": {"decomposition": "critic"},
            "includes 'state', but `state_decomposition_cfg` is None": {"decomposition": "state+reward"},
            "includes 'reward', but `reward_decomposition_cfg` is None": {"reward_decomposition_cfg": None},
            "needs a list of reward-term patterns": {"reward_decomposition_cfg": {"groups": {"feet": {"terms": []}}}},
            "Unknown keys in the reward group": {
                "reward_decomposition_cfg": {"groups": {"feet": {"terms": ["*"], "scale": 2.0}}}
            },
        }
        for message, update in misconfigurations.items():
            cfg = make_cfg()
            cfg["moe_cfg"].update(update)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_algorithm(env, cfg)

        algorithm = build_algorithm(env, make_cfg())
        obs = env.get_observations()
        algorithm.act(obs)
        zeros = torch.zeros(NUM_ENVS)
        with self.assertRaisesRegex(KeyError, "reward_terms"):
            algorithm.process_env_step(obs, zeros, zeros, {})
        with self.assertRaisesRegex(ValueError, "do not match any reward term"):
            algorithm.process_env_step(obs, zeros, zeros, {"reward_terms": {"lin_vel_tracking": zeros}})

        cfg = make_cfg()
        groups = {"feet": {"terms": ["feet_*"]}, "rest": {"terms": ["lin_*"]}}
        cfg["moe_cfg"]["reward_decomposition_cfg"]["groups"] = groups
        algorithm = build_algorithm(env, cfg)
        algorithm.act(obs)
        with self.assertRaisesRegex(ValueError, "'base_height' does not match any pattern"):
            extras = {"reward_terms": {**env.reward_terms(), "base_height": zeros}}
            algorithm.process_env_step(obs, zeros, zeros, extras)

    def test_warns_when_terms_do_not_sum_to_the_reward(self) -> None:
        env = GroupedRewardEnv()
        algorithm = build_algorithm(env, make_cfg())
        obs = env.get_observations()
        algorithm.act(obs)
        extras = {"reward_terms": {name: torch.ones(NUM_ENVS) for name in env.reward_terms()}}
        with self.assertWarnsRegex(UserWarning, "do not sum"):
            algorithm.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)

    def test_runner_learn_checkpoint_and_exports_without_moe(self) -> None:
        env = GroupedRewardEnv()
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = MoEOnPolicyRunner(env, make_cfg(), log_dir=temp_dir, device="cpu")
            runner.learn(num_learning_iterations=2)
            self.assertIsInstance(runner.alg, PPO)
            self.assertIsInstance(runner.alg.critic, RewardDecompositionCritic)

            checkpoint = Path(temp_dir) / "roundtrip.pt"
            runner.save(str(checkpoint))
            critic_state = {name: value.clone() for name, value in runner.alg.critic.state_dict().items()}
            with torch.no_grad():
                next(runner.alg.critic.parameters()).add_(1.0)
            runner.load(str(checkpoint))
            for name, value in runner.alg.critic.state_dict().items():
                torch.testing.assert_close(value, critic_state[name])

            runner.export_policy_to_jit(temp_dir)
            runner.export_policy_to_onnx(temp_dir)
            self.assertTrue((Path(temp_dir) / "policy.pt").is_file())
            self.assertTrue((Path(temp_dir) / "policy.onnx").is_file())


if __name__ == "__main__":
    unittest.main()

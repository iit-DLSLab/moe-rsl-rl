# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from moe_rsl_rl.modules import MoEModel


class PPO(RslRlPPO):
    """RSL-RL v5.4.2 PPO with MoE model construction and auxiliary routing losses."""

    def update(self) -> dict[str, float]:
        """Run PPO optimization, including optional MoE routing losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (  # type: ignore
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1e-8)

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if getattr(self._raw_actor, "use_log_prob_kl", False):
                        old_actions_log_prob = batch.old_actions_log_prob.squeeze(-1)
                        if old_actions_log_prob.shape != actions_log_prob.shape:
                            old_actions_log_prob = old_actions_log_prob.reshape_as(actions_log_prob)
                        kl = old_actions_log_prob - actions_log_prob.detach()
                    else:
                        kl = self.actor.get_kl_divergence(  # type: ignore
                            batch.old_distribution_params, distribution_params
                        )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = (  # type: ignore
                self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None
            )

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # MoE routing losses
            moe_models = [model for model in (self._raw_actor, self._raw_critic) if isinstance(model, MoEModel)]
            gate_models = [model for model in moe_models if model.use_gate_loss]
            if gate_models:
                weights = sum(
                    (model.mlp._last_gate_weights for model in gate_models),
                    start=torch.zeros_like(gate_models[0].mlp._last_gate_weights),
                )
                gate_entropy = -(weights * torch.log(weights + 1.0e-8)).sum(dim=-1).mean()
                loss = loss - 1.0e-4 * gate_entropy
            balance_models = [model for model in moe_models if model.use_load_balance_loss]
            if balance_models:
                loss = loss + 1.0e-4 * sum(model.load_balance_loss() for model in balance_models)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load a v5 checkpoint or migrate the model weights from the former v3 checkpoint layout."""
        if "model_state_dict" not in loaded_dict:
            return super().load(loaded_dict, load_cfg, strict)

        actor_state, critic_state = self._split_legacy_policy_state(loaded_dict["model_state_dict"])
        converted_dict = loaded_dict.copy()
        converted_dict["actor_state_dict"] = actor_state
        converted_dict["critic_state_dict"] = critic_state

        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": True,
            }
        else:
            load_cfg = load_cfg.copy()
            load_cfg["optimizer"] = False

        warnings.warn(
            "Loading model weights from an RSL-RL v3 checkpoint. The PPO optimizer is reinitialized because its "
            "parameter layout changed in RSL-RL v5.",
            stacklevel=2,
        )
        return super().load(converted_dict, load_cfg, strict)

    @staticmethod
    def _split_legacy_policy_state(state_dict: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        """Convert a monolithic v3 ActorCriticMoE state into separate v5 actor and critic states."""
        actor_state: dict[str, torch.Tensor] = {}
        critic_state: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            if name.startswith("actor."):
                actor_state[f"mlp.{name.removeprefix('actor.')}"] = value
            elif name.startswith("critic."):
                critic_state[f"mlp.{name.removeprefix('critic.')}"] = value
            elif name.startswith("actor_obs_normalizer."):
                suffix = name.removeprefix("actor_obs_normalizer.")
                actor_state[f"obs_normalizer.{suffix}"] = value
            elif name.startswith("critic_obs_normalizer."):
                suffix = name.removeprefix("critic_obs_normalizer.")
                critic_state[f"obs_normalizer.{suffix}"] = value
            elif name == "std":
                actor_state["distribution.std_param"] = value
            elif name == "log_std":
                actor_state["distribution.log_std_param"] = value
        return actor_state, critic_state

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct PPO using MoE models on the configured actor and/or critic side."""
        # Resolve class callables
        algorithm_name = cfg["algorithm"].pop("class_name")
        alg_class: type[PPO] = PPO if algorithm_name == "PPO" else resolve_callable(algorithm_name)  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Resolve the MoE configuration and its backwards-compatible expert-output aliases
        moe_cfg = cfg["moe_cfg"].copy()
        moe_cfg.pop("class_name", None)
        who = moe_cfg.pop("who")
        if who not in {"actor", "critic", "actor+critic"}:
            raise ValueError("`moe_cfg.who` must be 'actor', 'critic', or 'actor+critic'.")
        expert_output_dims = moe_cfg.pop("expert_output_dims", None)
        if expert_output_dims is None:
            expert_output_dims = moe_cfg.pop("expert_action_dims", None)
        if expert_output_dims is None:
            expert_output_dims = moe_cfg.pop("num_outputs_per_expert", None)
        if expert_output_dims is not None and "actor" not in who:
            raise ValueError("`expert_output_dims` can be used only when `who` includes 'actor'.")
        if moe_cfg.get("use_gaussian_mixture", False) and "actor" not in who:
            raise ValueError("`use_gaussian_mixture=True` requires `who` to include 'actor'.")

        # Initialize the actor
        if "actor" in who:
            actor: MLPModel = MoEModel(
                obs,
                cfg["obs_groups"],
                "actor",
                env.num_actions,
                **cfg["actor"],
                **moe_cfg,
                expert_output_dims=expert_output_dims,
            ).to(device)
        else:
            actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")

        # Initialize the critic
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            raise ValueError("`share_cnn_encoders` is not supported by MoE models.")
        if "critic" in who:
            critic_moe_cfg = moe_cfg.copy()
            critic_moe_cfg["use_gaussian_mixture"] = False
            critic: MLPModel = MoEModel(
                obs,
                cfg["obs_groups"],
                "critic",
                1,
                **cfg["critic"],
                **critic_moe_cfg,
            ).to(device)
        else:
            critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize storage and algorithm
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

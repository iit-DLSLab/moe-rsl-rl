# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import fnmatch
import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_class, resolve_obs_groups

from moe_rsl_rl.modules import MoEModel, RewardDecompositionCritic
from moe_rsl_rl.storage import RewardDecompositionRolloutStorage


class PPO(RslRlPPO):
    """RSL-RL v5.5.1 PPO with optional state decomposition (MoE models) and reward decomposition (multi-critic).

    With the reward decomposition, the environment publishes its reward terms every step as
    ``extras["reward_terms"] = {term_name: tensor(num_envs)}``, and ``reward_groups`` maps each group name to fnmatch
    patterns of term names. Each term goes to the first group with a matching pattern. Each group gets its own critic
    network and its own GAE. The actor is trained on the weighted sum of the per-group advantages, optionally
    normalized per group first.
    """

    def __init__(
        self,
        *args,
        use_pcgrad: bool = False,
        log_gradient_conflict: bool = False,
        reward_groups: dict[str, list[str]] | None = None,
        reward_group_weights: list[float] | None = None,
        normalize_group_advantages: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Reward decomposition: one critic, one reward channel, and one advantage weight per reward group
        self.reward_groups = dict(reward_groups) if reward_groups else None
        self.reward_group_names = list(self.reward_groups) if self.reward_groups else None
        self.normalize_group_advantages = normalize_group_advantages
        if self.reward_group_names is not None:
            num_groups = len(self.reward_group_names)
            if not isinstance(self._raw_critic, RewardDecompositionCritic) or (
                self._raw_critic.group_names != self.reward_group_names
            ):
                raise ValueError("The reward decomposition requires one critic per reward group.")
            storage = self.storage
            if not isinstance(storage, RewardDecompositionRolloutStorage) or storage.num_critics != num_groups:
                raise ValueError("The reward decomposition requires a rollout storage with one channel per group.")
            if self.rnd:
                raise ValueError("The reward decomposition does not support RND.")
            weights = [1.0] * num_groups if reward_group_weights is None else list(reward_group_weights)
            if len(weights) != num_groups:
                raise ValueError("`reward_group_weights` must contain one weight per reward group.")
            self.reward_group_weights = torch.tensor(weights, dtype=torch.float, device=self.device)
            self._checked_reward_group_sum = False
            # Resolved at the first step, when the environment's reward terms are known
            self._reward_term_names: list[str] | None = None
            self._reward_term_to_group: torch.Tensor | None = None

        self.use_pcgrad = use_pcgrad
        self.log_gradient_conflict = log_gradient_conflict
        if use_pcgrad or log_gradient_conflict:
            what = "PCGrad" if use_pcgrad else "Gradient-conflict logging"
            models = self._moe_models()
            if not models or any(
                not model.mlp.use_explicit_expert
                or not (model.mlp.use_shared_backbone or model.mlp.use_shared_backbone_and_head)
                for model in models
            ):
                raise ValueError(f"{what} requires explicit routing and a shared backbone on each MoE model.")
            if use_pcgrad and self.is_multi_gpu:
                raise ValueError("PCGrad currently supports single-device training only.")
            if self.actor.is_recurrent or self.critic.is_recurrent:
                raise ValueError(f"{what} currently supports feed-forward models only.")

    @property
    def use_reward_decomposition(self) -> bool:
        """Whether the critic is split into one network per reward group."""
        return self.reward_group_names is not None

    def _critic_models(self) -> list[nn.Module]:
        """Return the critic networks: one per reward group, or the single critic."""
        if isinstance(self._raw_critic, RewardDecompositionCritic):
            return list(self._raw_critic.critics.values())
        return [self._raw_critic]

    def _moe_models(self) -> list[MoEModel]:
        """Return every MoE network among the actor and the critic(s)."""
        return [model for model in (self._raw_actor, *self._critic_models()) if isinstance(model, MoEModel)]

    def _resolve_reward_terms(self, term_names: list[str]) -> None:
        """Assign every reward term to the first reward group with a matching fnmatch pattern."""
        group_terms: dict[str, list[str]] = {name: [] for name in self.reward_group_names}  # type: ignore[union-attr]
        term_to_group = []
        for term in term_names:
            for group_index, (name, patterns) in enumerate(self.reward_groups.items()):  # type: ignore[union-attr]
                if any(fnmatch.fnmatchcase(term, pattern) for pattern in patterns):
                    group_terms[name].append(term)
                    term_to_group.append(group_index)
                    break
            else:
                raise ValueError(
                    f"The reward term '{term}' does not match any pattern in `reward_decomposition_cfg.groups`."
                )
        empty_groups = [name for name, terms in group_terms.items() if not terms]
        if empty_groups:
            raise ValueError(f"The reward groups {empty_groups} do not match any reward term of the environment.")
        print("Reward decomposition, reward terms of each critic:")
        for name, terms in group_terms.items():
            print(f"\t{name}: {terms}")
        self._reward_term_names = term_names
        self._reward_term_to_group = torch.tensor(term_to_group, dtype=torch.long, device=self.device)

    def _stack_reward_groups(self, rewards: torch.Tensor, extras: dict) -> torch.Tensor:
        """Sum the reward terms published by the environment into a ``(num_envs, num_groups)`` tensor."""
        reward_terms = extras.get("reward_terms")
        if reward_terms is None:
            raise KeyError(
                "The reward decomposition expects the environment to publish extras['reward_terms'] = {term: rewards}."
            )
        if self._reward_term_names is None or reward_terms.keys() != set(self._reward_term_names):
            self._resolve_reward_terms(list(reward_terms))
        terms = torch.stack(
            [reward_terms[name].to(self.device) for name in self._reward_term_names], dim=-1  # type: ignore[union-attr]
        )
        # Exact sum of the terms of each group (a matmul would lose precision with TF32 enabled)
        group_rewards = torch.zeros(
            terms.shape[0], len(self.reward_group_names), device=self.device  # type: ignore[arg-type]
        ).index_add_(1, self._reward_term_to_group, terms.float())  # type: ignore[arg-type]
        # One-time check that the groups partition the reward the environment reports (and the logger shows)
        if not self._checked_reward_group_sum:
            self._checked_reward_group_sum = True
            if not torch.allclose(group_rewards.sum(dim=-1), rewards.view(-1).to(group_rewards), rtol=1e-4, atol=1e-5):
                warnings.warn(
                    "The reward terms do not sum to the environment reward: the critics are trained on a different"
                    " reward than the one that is logged.",
                    stacklevel=2,
                )
        return group_rewards

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step, storing one reward per reward group with the reward decomposition."""
        if not self.use_reward_decomposition:
            super().process_env_step(obs, rewards, dones, extras)
            return

        self.transition.rewards = self._stack_reward_groups(rewards, extras)
        self.transition.dones = dones
        # Bootstrapping on time outs, separately for every critic
        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * self.transition.values * time_outs  # type: ignore[operator]

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute per-group returns and the combined advantage with the reward decomposition."""
        if not self.use_reward_decomposition:
            super().compute_returns(obs)
            return

        st = self.storage
        last_values = self.critic(obs).detach()
        # GAE for every reward group at once: rewards and values have shape (num_envs, num_groups)
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.group_rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        # Combine the group advantages into the single advantage used by the actor. Without per-group normalization
        # and with unit weights, this equals the standard GAE of the total reward with V = sum of the critics.
        group_advantages = st.returns - st.values
        if self.normalize_group_advantages:
            group_advantages = (group_advantages - group_advantages.mean(dim=(0, 1))) / (
                group_advantages.std(dim=(0, 1)) + 1e-8
            )
        st.advantages = (group_advantages * self.reward_group_weights).sum(dim=-1, keepdim=True)
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    @staticmethod
    def _project_conflicting_gradients(gradients: torch.Tensor) -> torch.Tensor:
        """Return summed PCGrad gradients, projecting against the original task gradients."""
        projected = gradients.clone()
        for i in range(len(gradients)):
            for j in torch.randperm(len(gradients)).tolist():
                if i == j:
                    continue
                reference = gradients[j]
                dot = torch.dot(projected[i], reference)
                scale = dot.clamp(max=0) / reference.square().sum().clamp_min(1e-12)
                projected[i] -= scale * reference
        return projected.sum(dim=0)

    @staticmethod
    def _expert_backbone_gradients(
        model: MoEModel, sample_losses: torch.Tensor, expert_ids: torch.Tensor
    ) -> tuple[list[nn.Parameter], torch.Tensor | None]:
        """Return the shared-backbone parameters and one flattened backbone gradient per routed expert."""
        parameters = [parameter for parameter in model.mlp.shared_backbone.parameters() if parameter.requires_grad]
        if not parameters:
            return parameters, None
        gradients = []
        for expert_id in expert_ids.unique():
            # Sum / batch size preserves the original contribution of each expert's samples.
            task_loss = sample_losses[expert_ids == expert_id].sum() / sample_losses.numel()
            task_gradients = torch.autograd.grad(task_loss, parameters, retain_graph=True)
            gradients.append(torch.cat([gradient.reshape(-1) for gradient in task_gradients]))
        return parameters, torch.stack(gradients)

    def _pcgrad_correction(
        self, model: MoEModel, sample_losses: torch.Tensor, expert_ids: torch.Tensor
    ) -> list[tuple[nn.Parameter, torch.Tensor]]:
        """Compute a backbone gradient correction while retaining the ordinary PPO weighting."""
        parameters, stacked = self._expert_backbone_gradients(model, sample_losses, expert_ids)
        if stacked is None:
            return []
        return self._pcgrad_correction_from_gradients(parameters, stacked)

    def _pcgrad_correction_from_gradients(
        self, parameters: list[nn.Parameter], stacked: torch.Tensor
    ) -> list[tuple[nn.Parameter, torch.Tensor]]:
        correction = self._project_conflicting_gradients(stacked) - stacked.sum(dim=0)
        return list(zip(parameters, correction.split([parameter.numel() for parameter in parameters])))

    @classmethod
    @torch.no_grad()
    def _gradient_conflict_stats(cls, gradients: torch.Tensor) -> dict[str, float] | None:
        """Measure how much the per-expert backbone gradients conflict.

        - ``cos_mean`` / ``cos_min``: mean and minimum pairwise cosine similarity between expert gradients.
        - ``conflict_frac``: fraction of expert pairs with negative cosine similarity (the pairs PCGrad projects).
        - ``pcgrad_change``: ``||PCGrad(g) - sum(g)|| / ||sum(g)||``, i.e. how much PCGrad would alter the update.
        """
        if len(gradients) < 2:
            return None
        gradients = gradients.float()
        unit = gradients / gradients.norm(dim=1, keepdim=True).clamp_min(1e-12)
        i, j = torch.triu_indices(len(gradients), len(gradients), offset=1, device=gradients.device)
        cos = (unit @ unit.T)[i, j]
        total = gradients.sum(dim=0)
        change = (cls._project_conflicting_gradients(gradients) - total).norm() / total.norm().clamp_min(1e-12)
        return {
            "cos_mean": cos.mean().item(),
            "cos_min": cos.min().item(),
            "conflict_frac": (cos < 0).float().mean().item(),
            "pcgrad_change": change.item(),
        }

    def update(self) -> dict[str, float]:
        """Run PPO optimization, including optional MoE routing losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Per-critic value losses
        mean_group_value_losses = 0 if self.use_reward_decomposition else None
        # Gradient-conflict statistics between experts on the shared backbones
        conflict_sums: dict[str, float] = {}
        conflict_counts: dict[str, int] = {}

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch_idx, batch in enumerate(generator):
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

            # Optionally use mixed precision for the forward pass and loss computation (RSL-RL v5.5.1)
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
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
                surrogate_samples = torch.max(surrogate, surrogate_clipped)
                surrogate_loss = surrogate_samples.mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_samples = torch.max(value_losses, value_losses_clipped)
                else:
                    value_samples = (batch.returns - values).pow(2)
                # Sum over critics, so that each critic receives the gradient of a standalone critic (one critic: mean)
                value_loss = value_samples.sum(dim=-1).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                corrections = []
                # Without PCGrad, the extra backward passes for the conflict statistics run on the first mini-batch
                # only; with PCGrad, the per-expert gradients are computed anyway and every mini-batch is measured.
                measure_conflict = self.log_gradient_conflict and (self.use_pcgrad or batch_idx == 0)
                if self.use_pcgrad or measure_conflict:
                    critic_models = self._critic_models()
                    critic_names = (
                        [f"critic_{name}" for name in self.reward_group_names]  # type: ignore[union-attr]
                        if self.use_reward_decomposition
                        else ["critic"] * len(critic_models)
                    )
                    for model_name, model, sample_losses in (
                        ("actor", self._raw_actor, surrogate_samples),
                        *(
                            (critic_names[k], critic, self.value_loss_coef * value_samples[..., k].flatten())
                            for k, critic in enumerate(critic_models)
                        ),
                    ):
                        if not isinstance(model, MoEModel):
                            continue
                        expert_ids = model.mlp._last_gate_weights.squeeze(1).argmax(dim=-1)
                        if model is self._raw_actor:
                            # Entropy is evaluated only on original samples when using augmentation.
                            entropy_samples = torch.cat((entropy, entropy.new_zeros(len(sample_losses) - len(entropy))))
                            sample_losses = sample_losses - (
                                self.entropy_coef * entropy_samples * len(sample_losses) / len(entropy)
                            )
                        parameters, stacked = self._expert_backbone_gradients(model, sample_losses, expert_ids)
                        if stacked is None:
                            continue
                        if measure_conflict:
                            stats = self._gradient_conflict_stats(stacked)
                            for stat_name, value in (stats or {}).items():
                                key = f"grad_conflict/{model_name}/{stat_name}"
                                conflict_sums[key] = conflict_sums.get(key, 0.0) + value
                                conflict_counts[key] = conflict_counts.get(key, 0) + 1
                        if self.use_pcgrad:
                            corrections.extend(self._pcgrad_correction_from_gradients(parameters, stacked))

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
                moe_models = self._moe_models()
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
            # Preserve expert heads and auxiliary-loss gradients; change only backbone PPO gradients.
            for parameter, correction in corrections:
                parameter.grad.add_(correction.view_as(parameter))
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            # Clip every critic network on its own, as if it were trained alone
            for critic in self._critic_models():
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            if mean_group_value_losses is not None:
                mean_group_value_losses += value_samples.detach().flatten(0, -2).mean(dim=0)
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Update the normalizers. Since RSL-RL v5.5.1 this happens here, over the whole collected rollout, and no
        # longer at every environment step in process_env_step
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

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
        if mean_group_value_losses is not None:
            for name, group_loss in zip(self.reward_group_names, (mean_group_value_losses / num_updates).tolist()):  # type: ignore[arg-type]
                loss_dict[f"value_{name}"] = group_loss

        for key, value in conflict_sums.items():
            loss_dict[key] = value / conflict_counts[key]

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load a v5 checkpoint or migrate the model weights from the former v3 checkpoint layout."""
        # A v5 checkpoint already has separate actor/critic state dicts; delegate directly
        if "model_state_dict" not in loaded_dict:
            return super().load(loaded_dict, load_cfg, strict)

        # A v3 checkpoint stores one monolithic ActorCriticMoE state; split it into the v5 layout
        actor_state, critic_state = self._split_legacy_policy_state(loaded_dict["model_state_dict"])
        converted_dict = loaded_dict.copy()
        converted_dict["actor_state_dict"] = actor_state
        converted_dict["critic_state_dict"] = critic_state

        # The optimizer's parameter layout changed between v3 and v5, so it cannot be restored
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
        # Re-key every v3 parameter under its v5 module path (actor/critic MLPModel wraps the MoE in `mlp.`)
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
        # Resolve class callables and configs. As in RSL-RL v5.5.1, the sub-configs are deep copies with
        # ``class_name`` removed, so ``cfg`` keeps its class names (the runner and the logger read them).
        algorithm_name = cfg["algorithm"]["class_name"]
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        if algorithm_name == "PPO":
            alg_class = PPO
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Resolve the optional moe-rsl-rl configuration: the state decomposition (mixture of experts on the actor
        # and/or critic) and the reward decomposition (one critic per group of reward terms)
        expert_cfg, reward_decomposition = PPO._resolve_moe_cfg(cfg.get("moe_cfg"))
        use_pcgrad = expert_cfg.pop("use_pcgrad", False)
        if expert_cfg.pop("log_gradient_conflict", False):
            alg_cfg["log_gradient_conflict"] = True
        who = expert_cfg.pop("who", "")
        use_moe = bool(who)
        expert_output_dims = expert_cfg.pop("expert_output_dims", None)
        if expert_output_dims is None:
            expert_output_dims = expert_cfg.pop("expert_action_dims", None)
        if expert_output_dims is None:
            expert_output_dims = expert_cfg.pop("num_outputs_per_expert", None)
        if expert_output_dims is not None and "actor" not in who:
            raise ValueError("`expert_output_dims` can be used only when `who` includes 'actor'.")
        if expert_cfg.get("use_gaussian_mixture", False) and "actor" not in who:
            raise ValueError("`use_gaussian_mixture=True` requires `who` to include 'actor'.")

        # Initialize the actor
        if "actor" in who:
            actor: MLPModel = MoEModel(
                obs,
                cfg["obs_groups"],
                "actor",
                env.num_actions,
                **actor_cfg,
                **expert_cfg,
                expert_output_dims=expert_output_dims,
            ).to(device)
        else:
            actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")

        # Initialize the critic(s)
        if alg_cfg.pop("share_cnn_encoders", None):
            if use_moe or reward_decomposition is not None:
                raise ValueError("`share_cnn_encoders` is not supported by the state or reward decomposition.")
            critic_cfg["cnns"] = actor.cnns
        critic_expert_cfg = {**expert_cfg, "use_gaussian_mixture": False}

        def build_critic(obs_set: str) -> MLPModel:
            if "critic" in who:
                return MoEModel(obs, cfg["obs_groups"], obs_set, 1, **critic_cfg, **critic_expert_cfg).to(device)
            return critic_class(obs, cfg["obs_groups"], obs_set, 1, **critic_cfg).to(device)

        num_steps = cfg["num_steps_per_env"]
        if reward_decomposition is None:
            critic = build_critic("critic")
            storage = RolloutStorage("rl", env.num_envs, num_steps, obs, [env.num_actions], device)
        else:
            reward_groups, group_weights, normalize_group_advantages = reward_decomposition
            # A critic reads the observation set `critic_<group>` when it is configured, `critic` otherwise
            critic = RewardDecompositionCritic({
                name: build_critic(f"critic_{name}" if f"critic_{name}" in cfg["obs_groups"] else "critic")
                for name in reward_groups
            }).to(device)
            storage = RewardDecompositionRolloutStorage(
                len(reward_groups), env.num_envs, num_steps, obs, [env.num_actions], device
            )
            alg_cfg.update(
                reward_groups=reward_groups,
                reward_group_weights=group_weights,
                normalize_group_advantages=normalize_group_advantages,
            )
        print(f"Critic Model: {critic}")

        # Initialize the algorithm
        if use_pcgrad or alg_cfg.get("use_pcgrad", False):
            alg_cfg["use_pcgrad"] = True
        if alg_cfg.get("use_pcgrad", False) or alg_cfg.get("log_gradient_conflict", False):
            if cfg.get("torch_compile_mode") is not None:
                raise ValueError("PCGrad and gradient-conflict logging currently require torch_compile_mode=None.")
        alg = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    @staticmethod
    def _resolve_moe_cfg(
        moe_cfg: dict | None,
    ) -> tuple[dict, tuple[dict[str, list[str]], list[float], bool] | None]:
        """Split ``moe_cfg`` into the state and the reward decomposition.

        ``moe_cfg.decomposition`` selects ``"state"`` (mixture of experts, from ``state_decomposition_cfg``),
        ``"reward"`` (one critic per group of reward terms, from ``reward_decomposition_cfg``), or ``"state+reward"``.
        The sub-config that is not selected is ignored. Return the expert options (empty without state decomposition)
        and the reward groups (name -> fnmatch patterns), their advantage weights, and the normalization flag (``None``
        without reward decomposition).
        """
        if moe_cfg is None:
            return {}, None
        moe_cfg = dict(moe_cfg)
        moe_cfg.pop("class_name", None)
        decomposition = moe_cfg.pop("decomposition", None)
        state_cfg = moe_cfg.pop("state_decomposition_cfg", None)
        reward_cfg = moe_cfg.pop("reward_decomposition_cfg", None)
        if moe_cfg:
            raise ValueError(
                f"Unknown `moe_cfg` keys: {sorted(moe_cfg)}. The expert options go in `moe_cfg.state_decomposition_cfg`"
                " and the reward groups in `moe_cfg.reward_decomposition_cfg`."
            )
        parts = decomposition.split("+") if isinstance(decomposition, str) else []
        if not parts or len(set(parts)) != len(parts) or not set(parts) <= {"state", "reward"}:
            raise ValueError(
                f"`moe_cfg.decomposition` must be 'state', 'reward', or 'state+reward', not {decomposition!r}."
            )

        expert_cfg = {}
        if "state" in parts:
            if state_cfg is None:
                raise ValueError("`moe_cfg.decomposition` includes 'state', but `state_decomposition_cfg` is None.")
            expert_cfg = dict(state_cfg)
            expert_cfg.pop("class_name", None)
            if expert_cfg.get("who") not in {"actor", "critic", "actor+critic"}:
                raise ValueError("`state_decomposition_cfg.who` must be 'actor', 'critic', or 'actor+critic'.")

        reward_decomposition = None
        if "reward" in parts:
            if reward_cfg is None:
                raise ValueError("`moe_cfg.decomposition` includes 'reward', but `reward_decomposition_cfg` is None.")
            reward_decomposition = PPO._resolve_reward_decomposition_cfg(reward_cfg)
        return expert_cfg, reward_decomposition

    @staticmethod
    def _resolve_reward_decomposition_cfg(reward_cfg: dict) -> tuple[dict[str, list[str]], list[float], bool]:
        """Return the reward groups (name -> fnmatch patterns), their advantage weights, and the normalization flag."""
        reward_cfg = dict(reward_cfg)
        groups = reward_cfg.pop("groups", None)
        normalize_advantages = bool(reward_cfg.pop("normalize_advantages", False))
        if reward_cfg:
            raise ValueError(f"Unknown `reward_decomposition_cfg` keys: {sorted(reward_cfg)}.")
        if not groups:
            raise ValueError("`reward_decomposition_cfg.groups` must define at least one reward group.")

        patterns: dict[str, list[str]] = {}
        weights: list[float] = []
        for name, group in groups.items():
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError(f"Invalid reward group name: {name!r}.")
            group = dict(group)
            terms = group.pop("terms", None)
            weight = group.pop("weight", 1.0)
            if group:
                raise ValueError(f"Unknown keys in the reward group '{name}': {sorted(group)}.")
            terms = [terms] if isinstance(terms, str) else list(terms or [])
            if not terms or not all(isinstance(term, str) for term in terms):
                raise ValueError(f"The reward group '{name}' needs a list of reward-term patterns in `terms`.")
            patterns[name] = terms
            weights.append(float(weight))
        return patterns, weights, normalize_advantages

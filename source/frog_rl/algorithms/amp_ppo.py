# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) extension for PPO.

Ported from the classic NVIDIA AMP implementation (``frog_rl.algorithms.amp_ppo`` and
``frog_rl.runners.amp_on_policy_runner``) to the new ``frog_rl`` PPO API.

The AMP reward **replaces** the environment reward (with an optional task-reward lerp),
and the discriminator is trained with its own optimizer in a separate backward pass
(following the ``frog_rl.extensions`` pattern used by RND).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frog_rl.algorithms.ppo import PPO
from frog_rl.algorithms.amp_discriminator import AMPDiscriminator
from frog_rl.datasets import AMPLoader
from frog_rl.modules.normalization import EmpiricalNormalization
from frog_rl.storage.replay_buffer import ReplayBuffer
from frog_rl.utils import resolve_optimizer


def resolve_amp_config(alg_cfg: dict, obs, env) -> dict:
    """Resolve the AMP configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Observation dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        amp_cfg = alg_cfg["amp_cfg"]
        # Validate the amp_state observation group exists in the environment
        if "amp_state" not in obs:
            raise ValueError(
                f"The AMP configuration requires an 'amp_state' observation group, but it is not present in the "
                f"observations from the environment. Available observations: {list(obs.keys())}"
            )
        # Resolve the discriminator input dimension (state + next_state)
        state_dim = obs["amp_state"].shape[-1]
        amp_cfg["state_dim"] = state_dim
        amp_cfg["discriminator_input_dim"] = 2 * state_dim
        # Default the transition time to the environment step duration
        if "time_between_frames" not in amp_cfg:
            amp_cfg["time_between_frames"] = env.unwrapped.step_dt
        alg_cfg["amp_cfg"] = amp_cfg
    else:
        alg_cfg["amp_cfg"] = None
    return alg_cfg


class AMPPPO(PPO):
    """PPO with an AMP discriminator that shapes the reward and provides a discriminator loss."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        amp_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the AMP-PPO algorithm.

        Args:
            actor: The actor model.
            critic: The critic model.
            storage: The rollout storage.
            device: Device to place the models on.
            multi_gpu_cfg: Multi-GPU configuration.
            amp_cfg: AMP configuration dictionary.
            **kwargs: Additional keyword arguments forwarded to :class:`PPO`.

        Raises:
            ValueError: If ``amp_cfg`` is None.
        """
        super().__init__(actor, critic, storage, device=device, multi_gpu_cfg=multi_gpu_cfg, **kwargs)

        if amp_cfg is None:
            raise ValueError("The AMP configuration 'amp_cfg' is required for AMPPPO.")
        self.amp_cfg = amp_cfg

        # AMP observation state dimension
        self.state_dim = amp_cfg["state_dim"]

        # Discriminator
        self.discriminator = AMPDiscriminator(
            input_dim=amp_cfg["discriminator_input_dim"],
            amp_reward_coef=amp_cfg.get("amp_reward_coef", 1.0),
            hidden_layer_sizes=amp_cfg.get("amp_discr_hidden_dims", [1024, 512]),
            device=device,
            activation=amp_cfg.get("amp_discr_activation", "relu"),
            task_reward_lerp=amp_cfg.get("amp_task_reward_lerp", 0.0),
        )

        # Replay buffer storing the policy transitions (s, s_next)
        self.amp_storage = ReplayBuffer(
            self.state_dim, amp_cfg.get("amp_replay_buffer_size", 1000000), device
        )

        # Normalizer for the AMP states (updated from both policy and expert states)
        self.amp_normalizer = EmpiricalNormalization(shape=self.state_dim, until=int(1.0e8)).to(device)

        # Expert motion dataset
        self.amp_data = AMPLoader(
            device,
            time_between_frames=amp_cfg.get("time_between_frames", 0.02),
            preload_transitions=amp_cfg.get("amp_preload_transitions", True),
            num_preload_transitions=amp_cfg.get("amp_num_preload_transitions", 2000000),
            motion_files=amp_cfg["amp_motion_files"],
        )

        # Separate optimizer for the discriminator (with trunk/head weight decay like the classic AMP)
        disc_opt_class = resolve_optimizer(amp_cfg.get("discriminator_optimizer", "adam"))
        self.discriminator_optimizer = disc_opt_class(
            [
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": amp_cfg.get("amp_trunk_weight_decay", 1e-3),
                },
                {
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": amp_cfg.get("amp_head_weight_decay", 1e-2),
                },
            ],
            lr=amp_cfg.get("discriminator_lr", 1e-3),
        )

        # Weight of the gradient penalty
        self.grad_pen_coef = amp_cfg.get("grad_pen_coef", 10.0)

        # Current AMP state (set in act(), consumed in process_env_step)
        self._current_amp_state = None

    """
    Operations.
    """

    def act(self, obs) -> torch.Tensor:
        """Record the current AMP state and sample actions."""
        # Record the AMP state before the environment step
        self._current_amp_state = obs["amp_state"]
        return super().act(obs)

    def process_env_step(self, obs, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Replace the task reward with the AMP reward and store the policy transition."""
        next_amp_state = obs["amp_state"]

        # Account for terminal states so the (s, s_next) pair of the terminal step is valid
        if "terminal_amp_states" in extras:
            reset_env_ids = (dones > 0).flatten().nonzero(as_tuple=False).flatten()
            next_amp_state = next_amp_state.clone()
            next_amp_state[reset_env_ids] = extras["terminal_amp_states"][reset_env_ids]

        # Store the policy transition (s, s_next) in the replay buffer
        self.amp_storage.insert(self._current_amp_state, next_amp_state)

        # Compute the AMP reward and replace the task reward with it
        amp_reward, _ = self.discriminator.predict_amp_reward(
            self._current_amp_state, next_amp_state, rewards, normalizer=self.amp_normalizer
        )
        amp_reward = amp_reward.unsqueeze(1)

        super().process_env_step(obs, amp_reward, dones, extras)

        # Update the current AMP state for the next step
        self._current_amp_state = next_amp_state

    def update(self) -> dict[str, float]:
        """Run PPO updates followed by separate discriminator training steps."""
        # PPO update (actor, critic, and any extensions)
        loss_dict = super().update()

        # Train the discriminator on (policy transitions, expert transitions)
        mini_batch_size = (
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        if self.amp_storage.num_samples >= mini_batch_size:
            self._train_discriminator(mini_batch_size, loss_dict)

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for the policy and the discriminator."""
        super().train_mode()
        self.discriminator.train()
        self.amp_normalizer.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for the policy and the discriminator."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models and states for saving, including the discriminator."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        saved_dict["amp_normalizer"] = self.amp_normalizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the discriminator and AMP normalizer in addition to the policy models."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if "discriminator_optimizer_state_dict" in loaded_dict:
            self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        if "amp_normalizer" in loaded_dict:
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMP-PPO algorithm.

        Extends :meth:`PPO.construct_algorithm` by resolving the ``amp_state`` observation
        set and the AMP-specific configuration.
        """
        # Add the "amp_state" observation set so the discriminator state is resolved
        if cfg["algorithm"].get("amp_cfg") is not None and "amp_state" not in cfg["obs_groups"]:
            cfg["obs_groups"]["amp_state"] = ["amp_state"]
        # Resolve the AMP-specific configuration (discriminator input dim, transition time, etc.)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, env)
        return PPO.construct_algorithm(obs, env, cfg, device)

    def reduce_discriminator_parameters(self) -> None:
        """Average the discriminator gradients across all GPUs."""
        grads = [param.grad for param in self.discriminator.parameters() if param.grad is not None]
        all_grads = torch.cat([grad.view(-1) for grad in grads])
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    """
    Internal helpers.
    """

    def _train_discriminator(self, mini_batch_size: int, loss_dict: dict) -> None:
        """Train the discriminator on policy and expert transitions.

        Args:
            mini_batch_size: Number of transitions per mini-batch.
            loss_dict: Dictionary to accumulate the discriminator losses into.
        """
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        # Ensure the normalizer is in training mode so its statistics are updated
        self.amp_normalizer.train()

        amp_policy_generator = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
        amp_expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)

        for (policy_state, policy_next_state), (expert_state, expert_next_state) in zip(
            amp_policy_generator, amp_expert_generator
        ):
            # Normalize the states before feeding them to the discriminator
            policy_state = self.amp_normalizer(policy_state)
            policy_next_state = self.amp_normalizer(policy_next_state)
            expert_state = self.amp_normalizer(expert_state)
            expert_next_state = self.amp_normalizer(expert_next_state)

            # Discriminator logits: expert -> +1, policy -> -1
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss = nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=self.grad_pen_coef
            )

            loss = amp_loss + grad_pen_loss

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_discriminator_parameters()
            self.discriminator_optimizer.step()

            # Update the AMP normalizer statistics from the normalized states
            self.amp_normalizer.update(policy_state.detach())
            self.amp_normalizer.update(expert_state.detach())

            # Accumulate the statistics
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        # Average the discriminator losses and add them to the loss dictionary
        loss_dict["amp_loss"] = mean_amp_loss / num_updates
        loss_dict["amp_grad_pen"] = mean_grad_pen_loss / num_updates
        loss_dict["amp_policy_pred"] = mean_policy_pred / num_updates
        loss_dict["amp_expert_pred"] = mean_expert_pred / num_updates

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: BSD-3-Clause

"""Discriminator for AMP (Adversarial Motion Priors).

Ported from the classic NVIDIA AMP implementation (``frog_rl.algorithms.amp_discriminator``)
to the ``frog_rl.modules.MLP`` building block.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from frog_rl.modules import MLP
from frog_rl.modules.normalization import EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    """Discriminator that distinguishes expert motions from policy motions.

    The discriminator takes a concatenated state transition ``[state, next_state]`` and
    outputs a single logit ``d``. The AMP reward is computed from this logit as
    ``coef * max(0, 1 - 0.25 * (d - 1)^2)`` (quadratic, classic AMP).
    """

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float,
        hidden_layer_sizes: tuple[int, ...] | list[int],
        device: str,
        activation: str = "relu",
        task_reward_lerp: float = 0.0,
    ) -> None:
        """Initialize the discriminator.

        Args:
            input_dim: Dimension of the concatenated ``[state, next_state]`` input.
            amp_reward_coef: Scaling factor for the AMP reward.
            hidden_layer_sizes: Hidden dimensions of the MLP trunk.
            device: Device to place the model on.
            activation: Activation function of the MLP trunk.
            task_reward_lerp: Fraction of the task reward to mix into the AMP reward.
                Defaults to 0.0 (pure AMP reward).
        """
        super().__init__()

        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        # MLP trunk followed by a single linear head producing the logit
        self.trunk = MLP(input_dim, hidden_layer_sizes[-1], hidden_layer_sizes, activation=activation).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the discriminator logit for the input transitions."""
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0):
        """Compute the WGAN-GP gradient penalty on the expert transitions.

        The penalty enforces that the gradient of the discriminator output w.r.t. the
        expert input has a small norm, promoting smoothness of the discriminator.

        Args:
            expert_state: Expert states of shape ``(N, state_dim)``.
            expert_next_state: Expert next states of shape ``(N, state_dim)``.
            lambda_: Weight of the gradient penalty.

        Returns:
            The gradient penalty loss.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Enforce that the gradient norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer: EmpiricalNormalization | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the AMP reward from the discriminator logit.

        Args:
            state: Policy states of shape ``(N, state_dim)``.
            next_state: Policy next states of shape ``(N, state_dim)``.
            task_reward: Task reward of shape ``(N,)`` or ``(N, 1)`` (used only when
                ``task_reward_lerp > 0``).
            normalizer: Optional normalizer applied to the states before the forward pass.

        Returns:
            A tuple of the AMP reward of shape ``(N,)`` and the discriminator logit.
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer(state)
                next_state = normalizer(next_state)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward)
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r: torch.Tensor, task_r: torch.Tensor) -> torch.Tensor:
        """Mix the AMP reward with the task reward."""
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r.reshape(disc_r.shape)

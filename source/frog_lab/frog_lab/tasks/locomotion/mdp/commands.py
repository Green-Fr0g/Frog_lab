from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import frog_lab.tasks.locomotion.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommand(CommandTerm):
    """Command generator that assigns a discrete integer command to each environment.

    The command is sampled uniformly from a user-specified list of integers
    (e.g. ``[1, 2, 3]``). Each environment receives one of these values, which is
    used as the target command until the command is resampled.

    This is analogous to :class:`UniformVelocityCommand`, except that the command
    is a single discrete value drawn from :attr:`cfg.available_commands` instead of
    a continuous velocity in SE(2). The command tensor has shape ``(num_envs, 1)``.
    """

    cfg: DiscreteCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: DiscreteCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If :attr:`cfg.available_commands` is empty or contains
                non-integer values.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # sanity checks on the configuration
        if not self.cfg.available_commands:
            raise ValueError("The 'available_commands' list cannot be empty.")
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements of 'available_commands' must be integers.")

        # store the available commands as a tensor for efficient indexing
        self._available_commands = torch.tensor(
            self.cfg.available_commands, dtype=torch.int32, device=self.device
        )

        # create buffer to store the command: shape (num_envs, 1)
        self.command_b = torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        return (
            "DiscreteCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tAvailable commands: {self.cfg.available_commands}"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired discrete command. Shape is (num_envs, 1)."""
        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """No metrics are tracked for the discrete command."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new command uniformly from :attr:`cfg.available_commands`."""
        indices = torch.randint(len(self._available_commands), (len(env_ids),), device=self.device)
        self.command_b[env_ids] = self._available_commands[indices].unsqueeze(1)

    def _update_command(self):
        """The command does not require any post-processing."""
        pass


@configclass
class DiscreteCommandCfg(CommandTermCfg):
    """Configuration for the discrete command generator."""

    class_type: type = DiscreteCommand

    available_commands: list[int] = []
    """List of discrete integer values to sample from.

    Each environment is assigned one of these values as its command.
    Example: [1, 2, 3]
    """


class UniformScalarCommand(CommandTerm):
    """Command generator that generates a scalar command from a uniform distribution.

    Unlike the velocity command which generates a 3-DOF command in SE(2) (x-vel, y-vel,
    yaw-vel), this command generator produces a single value sampled uniformly from
    :attr:`cfg.ranges`. The command tensor has shape ``(num_envs, 1)``.
    """

    cfg: UniformScalarCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformScalarCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the lower bound of :attr:`cfg.ranges` exceeds its upper bound.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # sanity check on the configuration
        if self.cfg.ranges[0] > self.cfg.ranges[1]:
            raise ValueError(
                f"Invalid range for the scalar command: {self.cfg.ranges}. "
                "The lower bound must not exceed the upper bound."
            )

        # create buffer to store the command: shape (num_envs, 1)
        self.command_b = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        return (
            "UniformScalarCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tRange: {self.cfg.ranges}"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired scalar command. Shape is (num_envs, 1)."""
        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """No metrics are tracked for the scalar command."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new command uniformly from :attr:`cfg.ranges`."""
        r = torch.empty(len(env_ids), device=self.device)
        self.command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges)

    def _update_command(self):
        """The command does not require any post-processing."""
        pass


@configclass
class UniformScalarCommandCfg(CommandTermCfg):
    """Configuration for the uniform scalar command generator."""

    class_type: type = UniformScalarCommand

    ranges: tuple[float, float] = MISSING
    """Range within which the scalar command is sampled uniformly.

    Example: (-1.0, 1.0)
    """

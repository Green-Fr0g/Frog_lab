# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def phase_list(
    env: ManagerBasedRLEnv,
    outputs: Sequence[str] = ("sin", "cos"),
    cycle_time: float = 1.0
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    phase_val = env.episode_length_buf[:, None] * env.step_dt / cycle_time

    angle = 2 * torch.pi * phase_val
    components = {
        "sin": torch.sin(angle),
        "cos": torch.cos(angle),
        "phase": phase_val,
    }

    try:
        result_parts = [components[key] for key in outputs]
    except KeyError as exc:
        raise ValueError(f"Unsupported output key: {exc.args[0]}") from exc

    return torch.cat(result_parts, dim=-1)
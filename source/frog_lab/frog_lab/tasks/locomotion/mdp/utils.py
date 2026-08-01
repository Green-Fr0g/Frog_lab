"""Utility functions for terrain-aware operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_terrain_column_range(terrain_cfg, terrain_name: str, device) -> tuple[int, int] | None:
    """Helper function to calculate column range for a terrain type.

    Args:
        terrain_cfg: The terrain generator configuration.
        terrain_name: Name of the terrain.
        device: Torch device.

    Returns:
        Tuple of (col_start, col_end) or None if terrain not found.
    """
    if terrain_cfg.sub_terrains is None or terrain_name not in terrain_cfg.sub_terrains:
        return None

    # Line 30: sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())

    # Line 31: proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device)
    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=device
    )

    # Line 32: proportions = proportions / proportions.sum()
    proportions = proportions / proportions.sum()

    # Line 33: cumsum_props = torch.cumsum(proportions, dim=0)
    cumsum_props = torch.cumsum(proportions, dim=0)

    # Line 35: terrain_idx = sub_terrain_names.index(terrain_name)
    terrain_idx = sub_terrain_names.index(terrain_name)

    # Line 37: col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_start = round(
        (0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols
    )

    # Line 38: col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    # Line 40: return col_start, col_end
    return col_start, col_end


def is_env_assigned_to_terrain(env: ManagerBasedEnv, terrain_name: str) -> torch.Tensor:
    """Check which environments are initially assigned to the specified terrain type.

    Each environment is assigned to a specific terrain cell at initialization.
    This function returns a mask indicating which environments were assigned to the given terrain type.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").

    Returns:
        Boolean tensor of shape (num_envs,) where True means the environment is assigned to this terrain.
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator

    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    return (terrain.terrain_types >= col_start) & (terrain.terrain_types < col_end)


def is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Check which robots are currently standing on the specified terrain type.

    This function calculates which terrain grid cell each robot is on based on its world position,
    then checks if that cell's terrain type matches the specified terrain.

    Args:
        env: The environment instance.
        terrain_name: Name of the terrain to check (e.g., "pits", "stairs").
        asset_name: Name of the robot asset. Defaults to "robot".

    Returns:
        Boolean tensor of shape (num_envs,) where True means the robot is currently on this terrain.
    """
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator

    col_range = _get_terrain_column_range(terrain_cfg, terrain_name, env.device)
    if col_range is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    col_start, col_end = col_range

    asset = env.scene[asset_name]
    robot_pos_w = asset.data.root_pos_w[:, :2]

    terrain_origins = terrain.terrain_origins
    num_rows, num_cols, _ = terrain_origins.shape

    terrain_origins_2d = terrain_origins[:, :, :2].reshape(num_rows * num_cols, 2)

    distances = torch.cdist(robot_pos_w, terrain_origins_2d)
    closest_flat_idx = torch.argmin(distances, dim=1)

    col_idx = closest_flat_idx % num_cols

    return (col_idx >= col_start) & (col_idx < col_end)

"""Unitree G1 29-DOF velocity locomotion."""

import gymnasium as gym

from . import agents

gym.register(
    id="Rough-G1-29",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1_29DOFRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFRoughPPORunnerCfg",
    },
)

gym.register(
    id="Flat-G1-29",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1_29DOFFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1_29DOFFlatPPORunnerCfg",
    },
)

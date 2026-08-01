"""Flat-terrain velocity locomotion configuration for Unitree G1 29-DOF."""

from isaaclab.utils import configclass

from .rough_env_cfg import G1_29DOFRoughEnvCfg


@configclass
class G1_29DOFFlatEnvCfg(G1_29DOFRoughEnvCfg):
    """G1 29-DOF velocity locomotion on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        #state_Rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.lin_vel_z_l2.weight = -0.2

        #action_Rewards
        self.rewards.action_rate_l2.weight = -0.005

        #joint_Rewards
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]

        #task_Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        if self.__class__.__name__ == "G1_29DOFFlatEnvCfg":
            self.disable_zero_weight_rewards()

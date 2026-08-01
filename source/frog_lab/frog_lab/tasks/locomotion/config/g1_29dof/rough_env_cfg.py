"""Rough-terrain velocity locomotion configuration for Unitree G1 29-DOF."""

from isaaclab.utils import configclass

import frog_lab.tasks.locomotion.mdp as mdp
from frog_lab.assets.g1_29dof import G1_29DOF_CFG
from source.frog_lab.frog_lab.tasks.locomotion.locomotion_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class G1_29DOFRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """G1 29-DOF velocity locomotion on rough terrain."""

    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"

    # joint_names = [  # 29-DOF controllable joints, index: name
    #     "left_hip_pitch_joint",       # 0  L_LEG_HIP_PITCH
    #     "left_hip_roll_joint",        # 1  L_LEG_HIP_ROLL
    #     "left_hip_yaw_joint",         # 2  L_LEG_HIP_YAW
    #     "left_knee_joint",            # 3  L_LEG_KNEE
    #     "left_ankle_pitch_joint",     # 4  L_LEG_ANKLE_PITCH
    #     "left_ankle_roll_joint",      # 5  L_LEG_ANKLE_ROLL
    #     "right_hip_pitch_joint",      # 6  R_LEG_HIP_PITCH
    #     "right_hip_roll_joint",       # 7  R_LEG_HIP_ROLL
    #     "right_hip_yaw_joint",        # 8  R_LEG_HIP_YAW
    #     "right_knee_joint",           # 9  R_LEG_KNEE
    #     "right_ankle_pitch_joint",    # 10 R_LEG_ANKLE_PITCH
    #     "right_ankle_roll_joint",     # 11 R_LEG_ANKLE_ROLL
    #     "waist_yaw_joint",            # 12 WAIST_YAW
    #     "waist_roll_joint",           # 13 WAIST_ROLL
    #     "waist_pitch_joint",          # 14 WAIST_PITCH
    #     "left_shoulder_pitch_joint",  # 15 L_SHOULDER_PITCH
    #     "left_shoulder_roll_joint",   # 16 L_SHOULDER_ROLL
    #     "left_shoulder_yaw_joint",    # 17 L_SHOULDER_YAW
    #     "left_elbow_joint",           # 18 L_ELBOW
    #     "left_wrist_roll_joint",      # 19 L_WRIST_ROLL
    #     "left_wrist_pitch_joint",     # 20 L_WRIST_PITCH
    #     "left_wrist_yaw_joint",       # 21 L_WRIST_YAW
    #     "right_shoulder_pitch_joint", # 22 R_SHOULDER_PITCH
    #     "right_shoulder_roll_joint",  # 23 R_SHOULDER_ROLL
    #     "right_shoulder_yaw_joint",   # 24 R_SHOULDER_YAW
    #     "right_elbow_joint",          # 25 R_ELBOW
    #     "right_wrist_roll_joint",     # 26 R_WRIST_ROLL
    #     "right_wrist_pitch_joint",    # 27 R_WRIST_PITCH
    #     "right_wrist_yaw_joint",      # 28 R_WRIST_YAW
    # ]

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Observations
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None

        # Actions
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        # Events
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [f"^(?!.*{self.base_link_name}).*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # Rewards
        self.rewards.is_terminated.weight = -200.0

        #state_Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.78
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.stand_still.weight = -0.8

        #joints_Rewards
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [
            ".*_hip_.*",
            ".*_knee_joint",
            ".*_ankle_.*",
        ]
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]

        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_arms.params["asset_cfg"].joint_names =[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
        self.rewards.joint_deviation_hip.weight = -0.1
        self.rewards.joint_deviation_hip.params["asset_cfg"].joint_names =[".*_hip_yaw.", ".*_hip_roll."]
        self.rewards.joint_deviation_torso.weight = -0.1
        self.rewards.joint_deviation_torso.params["asset_cfg"].joint_names = ["waist_.*"]
        
        #action_Rewards
        self.rewards.action_rate_l2.weight = -0.005
 
        #feet_Rewards
        self.rewards.feet_clearance.weight = 1.0
        self.rewards.feet_clearance.params["target_height"] = 0.05
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0

        #other_Rewards
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        #tasks_Rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0

        if self.__class__.__name__ == "G1_29DOFRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # Terminations
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.base_height.params["minimum_height"] = 0.2

        # Curriculum
        self.curriculum.command_levels_lin_vel = (0.1, 1.0)
        self.curriculum.command_levels_ang_vel = (0.1, 1.0)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

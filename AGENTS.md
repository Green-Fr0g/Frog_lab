# frog_lab — 人形机器人训练框架

本项目是一个**人形机器人训练框架**,主要训练两种 Unitree 机器人:

- **G1** 29 自由度 (DOF)
- **G1** 23 自由度 (DOF)

## 相关项目目录

### 训练与仿真 (train/lab)

| 目录别名       | 路径                                   | 说明 |
| -------------- | -------------------------------------- | ---- |
| isaaclab       | `/home/zhang/code/train/lab/IsaacLab`       | |
| robot_lab      | `/home/zhang/code/train/lab/robot_lab`      | |
| beyond_mimic   | `/home/zhang/code/train/lab/whole_body_tracking` | |
| unitree_rl_lab | `/home/zhang/code/train/lab/unitree_rl_lab` | |

### 强化学习算法 (train/gym)

| 目录别名       | 路径                                   | 说明 |
| -------------- | -------------------------------------- | ---- |
| DreamWaQ       | `/home/zhang/code/train/gym/DreamWaQ`       | |
| AMP_for_hardware | `/home/zhang/code/train/gym/AMP_for_hardware` | |
| instinct_rl    | `/home/zhang/code/train/lab/instict/instinct_rl` | |
| amp_mjlab      | `/home/zhang/code/train/lab/AMP_mjlab`      | |

## 要求
不能删除和修改这个项目以外的文件以及相应的内容
因为电脑配置的原因，不可以启动isaacsim仿真进行验证
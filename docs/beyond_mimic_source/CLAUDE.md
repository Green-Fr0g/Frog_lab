# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在该仓库中工作提供指导。

## 项目概述

BeyondMimic 是一个基于 Isaac Lab (v2.1.0) / Isaac Sim (4.5.0) 的人形机器人运动跟踪训练框架。它使用 PPO 强化学习算法训练策略网络，让机器人跟踪参考运动，重点面向 Unitree G1 的 sim-to-real 部署。

### 核心依赖
- **Isaac Lab 2.1.0** — 强化学习框架（基于 manager 的 MDP 架构）
- **Isaac Sim 4.5.0** — NVIDIA Omniverse 物理仿真器
- **RSL-RL** — PPO 训练实现
- **Weights & Biases (W&B)** — 运动数据仓库、实验日志和模型存储

## 常用命令

### 安装
```bash
# 安装 Python 包（需要在 Isaac Lab 环境中执行）
python -m pip install -e source/whole_body_tracking
```

### 代码检查（pre-commit）
```bash
# 运行所有 pre-commit 检查（black, flake8, isort, codespell 等）
pre-commit run --all-files
```

### 运动数据预处理
```bash
# 将 CSV 运动数据转为 NPZ（含全身正运动学信息），并上传至 W&B 仓库
python scripts/csv_to_npz.py --input_file {motion}.csv --input_fps 30 --output_name {motion_name} --headless

# 在 Isaac Sim 中回放运动数据以验证预处理结果
python scripts/replay_npz.py --registry_name={org}-org/wandb-registry-motions/{motion_name}
```

### 训练
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {org}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project} --run_name {run_name}
```

### 评估
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
  --num_envs=2 --wandb_path={wandb-run-path}
```

### 环境变量
将 `WANDB_ENTITY` 设为组织名（**不是**个人用户名）。

## 代码架构

### 目录结构

```
source/whole_body_tracking/
├── whole_body_tracking/
│   ├── assets/                      # 机器人 URDF 文件 + 网格（通过 GCS 下载）
│   ├── robots/                      # 机器人配置
│   │   ├── g1.py                    #   Unitree G1：关节配置、执行器参数、动作缩放
│   │   ├── smpl.py                  #   SMPL 人体模型配置（仅仿真用）
│   │   └── actuator.py              #   DelayedImplicitActuator — 模拟执行器延迟
│   ├── tasks/tracking/
│   │   ├── tracking_env_cfg.py      #   基础环境配置：场景、MDP 全部设置
│   │   ├── mdp/                     #   MDP 原子函数（Isaac Lab manager API）
│   │   │   ├── commands.py          #     参考运动加载 + 自适应采样 + 重置
│   │   │   ├── rewards.py           #     DeepMimic 风格指数奖励函数
│   │   │   ├── observations.py      #     策略观测项（身体位姿、速度等）
│   │   │   ├── events.py            #     域随机化（关节默认位置、质心、摩擦）
│   │   │   └── terminations.py      #     早停条件（位置/朝向超限）
│   │   └── config/
│   │       ├── g1/                  #     G1 环境配置 + PPO 超参数
│   │       │   ├── flat_env_cfg.py
│   │       │   └── agents/rsl_rl_ppo_cfg.py
│   │       └── humanoid/            #     SMPL 人体环境配置 + PPO 超参数
│   │           ├── flat_env_cfg.py
│   │           └── agents/rsl_rl_ppo_cfg.py
│   └── utils/
│       ├── my_on_policy_runner.py   #   自定义 RSL-RL runner：保存时导出 ONNX + 关联 W&B
│       └── exporter.py              #   ONNX 导出 + 运动元数据（关节信息、动作缩放等）
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                 #   训练入口（Hydra + AppLauncher）
│   │   ├── play.py                  #   评估/回放入口（含 ONNX 导出）
│   │   └── cli_args.py              #   训练/评估脚本的通用参数解析
│   ├── csv_to_npz.py                #   运动预处理：CSV → NPZ（含正运动学计算）
│   ├── replay_npz.py                #   在 Isaac Sim 中回放运动以验证
│   └── upload_npz.py                #   上传 NPZ 到 W&B 仓库
├── config/extension.toml            # Isaac Lab 扩展元数据
├── pyproject.toml                   # 项目工具配置（isort, pyright）
├── .pre-commit-config.yaml          # pre-commit 钩子（black, flake8, isort, codespell, pyupgrade）
└── .flake8                          # flake8 配置（从 Isaac Lab 复制）
```

### MDP 架构（Manager-Based RL）

项目使用 Isaac Lab 的 **manager-based RL API**。环境配置类 `TrackingEnvCfg` 由多个 manager 配置组成，每个配置引用 `mdp/` 中的原子函数：

| Manager | 配置类 | 核心功能 |
|---|---|---|
| **场景 Scene** | `MySceneCfg` | 地面、机器人关节体、灯光、接触传感器 |
| **指令 Commands** | `CommandsCfg` → `MotionCommandCfg` | 参考运动加载、自适应失败偏向采样、初始状态随机化 |
| **动作 Actions** | `ActionsCfg` | 关节位置目标 + 动作缩放 |
| **观测 Observations** | `ObservationsCfg` | 策略观测（身体位姿、关节状态、动作）+ 特权观测（critic） |
| **奖励 Rewards** | `RewardsCfg` | 锚点/身体位姿和速度的指数误差奖励 + 正则项 |
| **终止 Terminations** | `TerminationsCfg` | 超时、锚点位置/朝向超限、末端执行器超限 |
| **事件 Events** | `EventCfg` | 域随机化：摩擦系数、质心、关节默认位置、外力推 |

### 运动跟踪流水线

1. **运动预处理** (`scripts/csv_to_npz.py`) — 将重定向后的 CSV 运动转为 NPZ 格式，通过正运动学计算全身位姿（位置、四元数、线速度、角速度、加速度），上传到 W&B 仓库。
2. **运动加载** (`mdp/commands.py` `MotionLoader`) — 从 W&B 工件加载 NPZ，按时间步索引提供 `joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`。
3. **自适应采样** (`MotionCommand._adaptive_sampling`) — 基于失败次数的自适应采样：将运动划分为多个时间窗口（bin），跟踪每个窗口的失败次数，用平滑核计算采样概率，优先训练困难的运动片段。
4. **策略训练** — PPO (RSL-RL)，Actor/Critic 网络结构 [512, 256, 128]，以跟踪误差奖励为优化目标。
5. **ONNX 导出** — 训练中（自定义 `MotionOnPolicyRunner`）和评估时（`play.py`）将策略导出为 ONNX 格式，附加上关节名称、刚度/阻尼、观测历史长度、动作缩放等元数据，供部署使用。

### 任务注册

Gym 环境通过 `tasks/__init__.py` 自动注册，它使用 `isaaclab_tasks.utils.import_packages` 扫描 `tasks/tracking/config/` 下的所有配置类。

已注册任务：
- `Tracking-Flat-G1-v0` — Unitree G1 平地跟踪（标准）
- `Tracking-Flat-G1-LowFreq-v0` — 低控制频率版本
- `Tracking-Flat-G1-WoStateEstimation-v0` — 无状态估计观测版本
- `Tracking-Flat-Humanoid-v0` — SMPL 人体模型跟踪
- `Tracking-Flat-Humanoid-Walk-v0`、`-WalkBack-v0`、`-WalkBox-v0` — SMPL 特定运动文件

### 机器人配置

机器人配置 (`robots/g1.py`) 定义了：
- **关节体（Articulation）** — URDF 路径、初始关节位置、碰撞设置
- **执行器分组** — 各关节组（腿部、脚部、腰部、手臂）根据电机型号（5020、7520_14、7520_22、4010）分别设置刚度、阻尼、转子惯量，由自然频率（10 Hz）和阻尼比（2.0）计算得到
- **动作缩放** — 自动计算为 `0.25 * 力矩上限 / 刚度`
- **延迟执行器** (`actuator.py`) — 可选的随机通信延迟模拟，通过环形缓存区实现，提升 sim-to-real 鲁棒性

### 关键设计模式

- **`@configclass`** — Isaac Lab 的配置类装饰器，将 dataclass 转为 OmegaConf 兼容配置，支持 `__post_init__` 后处理。子配置通过嵌套 `@configclass` 对象组合。
- **`MISSING`** — 哨兵值（`from dataclasses import MISSING`），标记使用前必须由子类覆盖的字段。
- **MDP 函数签名** — 所有奖励/观测/终止函数的签名遵循 `(env: ManagerBasedRLEnv, **params) -> torch.Tensor` 约定。
- **Hydra 任务配置** — 训练/评估脚本通过 `@hydra_task_config` 装饰器从注册中心按任务名加载环境配置和智能体配置。
- **AppLauncher** — 所有使用 Isaac Sim 的脚本必须在加载其他模块**之前**通过 `AppLauncher` 启动 Omniverse 应用。

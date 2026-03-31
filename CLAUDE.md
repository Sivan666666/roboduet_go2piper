# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RoboDuet is a PyTorch + Isaac Gym framework for whole-body legged loco-manipulation on quadruped robots (Go1/Go2) with robot arms (ARX5/Piper). It uses a cooperative policy mechanism with a two-stage training strategy (pretrained separate policies → unified policy) and supports zero-shot sim-to-real transfer.

Based on [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). Paper and project page at [locomanip-duet.github.io](https://locomanip-duet.github.io/).

## Environment Setup

Requires Python 3.8 (Isaac Gym constraint) and NVIDIA GPU:

```bash
conda create -n roboduet python=3.8
conda activate roboduet
# Install Isaac Gym Preview 4 first (from NVIDIA developer portal)
cd isaacgym/python && pip install -e .
# Then install this repo
cd RoboDuet
pip install -r requirements.txt
pip install -e .
```

Key dependencies: `params-proto==2.10.5` (config management), `wandb` (logging), `pytorch3d`, `pyzmq`/`lcm` (communication).

## Training & Inference Commands

```bash
# Train (two-stage: pretrained → unified)
python scripts/auto_train.py --num_envs 4096 --run_name test_roboduet --sim_device cuda:0 --robot go1 [--headless]

# Train unified stage only
python scripts/unified_train.py [same args]

# Play with keyboard
python scripts/play_by_key.py --logdir runs/<run_path> --ckptid <iter> --sim_device cuda:0

# VR control (3 terminals)
python scripts/vr_play/vr_streaming.py        # PC with VR headset
python scripts/vr_play/play_by_remote.py --logdir <path> --ckptid <iter>  # training machine
python scripts/vr_play/remote_pub.py           # training machine

# AprilTag visual control
python scripts/play_apriltag.py
```

Key training args: `--debug` (12 envs), `--no_wandb`, `--offline`, `--resume`, `--wo_two_stage`, `--use_rot6d`, `--seed`.

## Architecture: Two-Stage Cooperative Policy

### Environment Class Hierarchy

```
BaseTask → LeggedRobot → VelocityTrackingEasyEnv → HistoryWrapper
```

- `go1_gym/envs/automatic/legged_robot.py` — Core simulation: DOF management, rewards, domain randomization, resets
- `go1_gym/envs/automatic/__init__.py` — `VelocityTrackingEasyEnv`: arm/dog observation computation, body planning from arm output
- `go1_gym/envs/wrappers/history_wrapper.py` — Maintains 15-step observation history, returns dict with `obs`, `privileged_obs`, `obs_history`

### Two-Stage Training

**Stage 1 (iterations 0–10000): Pretrained separate policies** (`go1_gym_learn/ppo_cse_automatic/`)
- `DogActorCritic` (`dog_ac.py`): adaptation module + actor → 12 locomotion actions
- `ArmActorCritic` (`arm_ac.py`): adaptation module + history encoder + actor → 6 arm actions
- Each has its own PPO update and rollout storage

**Stage 2 (iterations 10000+): Unified policy** (`go1_gym_learn/ppo_cse_unified/`)
- `Unified2ActorCritic` (`unified2head_ac.py`): shared backbone with two output heads (12 dog + 6 arm)
- Enables gradient flow between locomotion and manipulation
- Transition controlled by sigmoid blending in `go1_gym/utils/global_switch.py`

### Global Switch (`go1_gym/utils/global_switch.py`)

Controls the pretrained→unified transition:
- `pretrained_to_hybrid_start` / `pretrained_to_hybrid_end` — iteration window for blending
- `get_reward_scales()` — blends reward scales during transition
- `get_beta()` — blending factor (0.0 → 0.5)

### Observation & Action Structure

- **Dog obs (56 dims)**: base angular/linear vel, roll/pitch/yaw, 12 joint pos, 12 joint vel, 5 commands
- **Arm obs (20 dims)**: 6 arm joint pos deltas, 6 actions, 6 arm commands, roll, pitch
- **Actions**: 18 total (12 locomotion + 6 arm), split across two policy heads

## Key Config Files

- `go1_gym/envs/automatic/legged_robot_config.py` — Master `Cfg` class (env, rewards, terrain, domain randomization, hybrid params, arm/dog settings)
- `go1_gym/envs/go1/go1_config.py` — Robot-specific defaults (Go1/Go2)
- `go1_gym/envs/go1/asset_config.py` — URDF asset paths
- Training hyperparams: `PPO_Args`, `DogAC_Args`, `ArmAC_Args`, `Unified2AC_Args`, `RunnerArgs` (defined in their respective modules)

## Output Structure

Training runs save to `runs/<run_name>/<date>/auto_train/<timestamp>/` with subdirectories: `checkpoints_arm/`, `checkpoints_dog/`, `deploy_model/`, `videos/`, `scripts/`, `params.txt`, `log.txt`.

## Robot Assets

URDF files in `resources/robots/`: `go2/urdf/arx5go2.urdf` (Go2+ARX5), `go2/urdf/go2piper.urdf` (Go2+Piper), plus backups for other configurations.

## Deployment

Separate repo: [RoboDuet-Deployment](https://github.com/locomanip-duet/RoboDuet_Deployment) for Unitree Go1/Go2 EDU with ARX5 and Meta Quest 3 VR control.

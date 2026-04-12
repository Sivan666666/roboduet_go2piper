# 机械臂 Reward 总结

这份文件总结的是当前 [`scripts/auto_train.py`](/home/sivan/whole_body/RoboDuet/scripts/auto_train.py) 这套训练配置下，和机械臂相关的 reward。

## 说明

- 当前策略步长 `dt = 0.02`
  - `control.decimation = 4`
  - `sim.dt = 0.005`
- 环境内部会先把非零 reward scale 乘上 `dt`，再参与每一步 reward 计算。
- 所以下面表里有两列：
  - `配置 scale`：你在配置文件或 `auto_train.py` 里直接设置的值
  - `单步实际 scale`：真正每一步生效的 scale，也就是 `配置 scale * 0.02`

## 当前启用的机械臂相关 Reward

| Reward 名字 | 作用 | 配置 scale | 单步实际 scale | 简单公式 |
|---|---|---:|---:|---|
| `arm_manip_commands_tracking_combine` | 机械臂末端位置 + 姿态跟踪主 reward | `1.0` | `0.02` | `exp(-(w_lpy * err_lpy + w_rpy * err_rpy))` |
| `vis_manip_commands_tracking_lpy` | 只看末端位置跟踪的 reward | `1.0` | `0.02` | `exp(-err_lpy)` |
| `vis_manip_commands_tracking_rpy` | 只看末端姿态跟踪的 reward | `1.0` | `0.02` | `exp(-err_rpy)` |
| `orientation_control` | 惩罚机身实际 roll/pitch 和期望机身姿态不一致 | `-10.0` | `-0.2` | `||g_proj_xy - g_proj_xy_des||^2` |
| `orientation_heuristic` | 当机身 pitch 和机械臂前后伸展方向共同变危险时的启发式惩罚 | `-2.0` | `-0.04` | `guide(pitch, delta_z)` |
| `arm_control_smoothness_1` | 惩罚 planner 输出变化过快 | `-0.1` | `-0.002` | `||plan_t - plan_{t-1}||^2` |
| `arm_control_limits` | 惩罚 planner 给出的机身 roll/pitch 超出允许范围 | `-5.0` | `-0.1` | `limit_violation(plan_actions)` |
| `arm_energy` | 惩罚机械臂能量/功率消耗过大 | `-0.00004` | `-0.0000008` | `sum((tau_arm * qdot_arm)^2)` |
| `arm_dof_acc` | 惩罚机械臂关节加速度过大 | `-2.5e-6` | `-5e-8` | `sum(((qdot_{t-1} - qdot_t)/dt)^2)` |
| `arm_action_rate` | 惩罚机械臂 action 变化太快 | `-0.1` | `-0.002` | `sum((a_t - a_{t-1})^2)` |

## 主跟踪 Reward 细节

### `arm_manip_commands_tracking_combine`

这是当前训练里最核心的一条机械臂 reward。

- 位置项：
  - `err_lpy = sum(abs(actual_lpy - target_lpy) / lpy_range)`
- 姿态项：
  - `err_rpy = sum(abs(actual_abg - target_abg) / rpy_range)`
- 合并后：
  - `reward = exp(-(w_lpy * err_lpy + w_rpy * err_rpy))`

你现在这版已经不是平方误差，而是：

- 先做绝对值
- 再按命令范围归一化
- 然后求和

## 当前这条主 Reward 的权重课程

这些设置来自 [`scripts/auto_train.py`](/home/sivan/whole_body/RoboDuet/scripts/auto_train.py)：

- `manip_weight_lpy_start = 4`
- `manip_weight_lpy_end = 3`
- `manip_weight_rpy_start = 0`
- `manip_weight_rpy_end = 1`
- `manip_weight_keep_sum_constant = True`
- `manip_weight_transition_iters = 5000`

意思是：

- 训练前期更偏重末端位置跟踪
- 训练后期逐渐加入姿态跟踪
- 总权重和保持不变，从 `4 + 0` 平滑过渡到 `3 + 1`

## 当前为 0 或未启用的机械臂项

下面这些 reward 函数虽然代码里有，但你现在这套训练配置下没有实际起作用，或者 scale 为 0：

| Reward 名字 | 当前状态 |
|---|---|
| `arm_dof_vel` | scale 为 0 |
| `arm_action_smoothness_1` | scale 为 0 |
| `arm_action_smoothness_2` | scale 为 0 |
| `arm_vel_control` | 当前没有对应非零 scale |
| `arm_orientation_control` | 当前没有对应非零 scale |

## 补充说明

有两个名字里带 `vis_` 的 reward：

- `vis_manip_commands_tracking_lpy`
- `vis_manip_commands_tracking_rpy`

这两个 reward 当前虽然会被计算，而且也会记录到 `episode_sums` 里用于统计/日志，但**不会**加入最终的 `rew_buf_arm` 或 `rew_buf_dog`，因此**不会参与 PPO 更新**。

原因是环境在汇总 reward 时，对这两个名字做了单独分支处理：

- 先计算 `rew = rew_raw * reward_scales[name]`
- 然后只执行 `self.episode_sums[name] += rew`
- 接着直接 `continue`

所以它们的作用更接近：

- 训练过程中的监控指标
- 日志统计项
- 便于观察末端位置/姿态跟踪情况

而不是实际驱动策略优化的 reward 项。

## 相关代码位置

- [`scripts/auto_train.py`](/home/sivan/whole_body/RoboDuet/scripts/auto_train.py)
- [`go1_gym/envs/rewards/rewards.py`](/home/sivan/whole_body/RoboDuet/go1_gym/envs/rewards/rewards.py)
- [`go1_gym/envs/automatic/legged_robot_config.py`](/home/sivan/whole_body/RoboDuet/go1_gym/envs/automatic/legged_robot_config.py)
- [`go1_gym/envs/automatic/legged_robot.py`](/home/sivan/whole_body/RoboDuet/go1_gym/envs/automatic/legged_robot.py)

# `my_test_v3` 相比上一个 Commit `ddfc082`（`VR适配 可以通信`）的具体改动

## 对比范围

- 上一个 commit：`ddfc082` `VR适配 可以通信`
- 当前 commit：`aaf1810` `my_test_v3`

说明：
- 这次提交**没有直接修改** `scripts/vr_play/play_by_remote.py`
- 但围绕 VR 使用时最关键的几件事做了增强：末端坐标系对齐、可视化对齐、目标采样安全、运行时调参、调试验证

## 1. 可视化更新时序调整，VR/手动控制时 marker 更贴近当前命令

涉及文件：
- `go1_gym/envs/automatic/__init__.py`

具体改动：
- 在 `KeyboardWrapper.render()` 里，先执行 `update_arm_commands()`，再画目标球、EE 坐标轴和 base 坐标轴
- 同时把 arm 目标碰撞框也接入了 viewer 绘制

效果：
- 解决“命令已经更新，但画面还是上一帧 marker”的错位感
- VR/键控调试时，看到的目标显示和当前实际命令更一致

## 2. 末端姿态显示改成使用工具坐标系偏置 `ee_rot_offset`

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`

具体改动：
- `_draw_ee_ori_coord()` 不再直接拿原始 `end_effector_state[..., 3:7]` 画姿态
- 改成先做：
  `ee_quat_new = quat_mul(ee_quat_raw, self.ee_rot_offset)`
- 然后用偏置后的姿态去画黄色末端球和姿态坐标轴

同时：
- `grasper_move` 从 `0.1` 改为 `0.12`
- 这段偏移也由偏置后的 EE 姿态来旋转

效果：
- viewer 里画出来的末端朝向，更接近真实工具坐标系
- 黄色球的位置也更像夹爪尖端，而不是 link 原点

## 3. 末端位置与姿态计算统一到“偏置后工具坐标系”

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`
- `go1_gym/envs/rewards/rewards.py`

具体改动：
- `get_lpy_in_base_coord()`：
  - 先对 EE 四元数乘 `ee_rot_offset`
  - 再用 `grasper_move = [0.12, 0, 0]` 计算尖端位置
- `get_alpha_beta_gamma_in_base_coord()`：
  - 也先对 EE 四元数乘 `ee_rot_offset`
  - 再转换到 base 坐标系求 `alpha/beta/gamma`

效果：
- 末端位置 `lpy`
- 末端姿态 `abg`
- viewer 里的 EE 坐标轴
- reward 使用的 EE 几何定义

这几套定义变得更一致，减少“显示对了但 reward 认为不对”的情况。

## 4. 新增 arm 目标安全采样，避免采到碰撞或插地目标

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`
- `go1_gym/envs/automatic/legged_robot_config.py`

新增配置：
- `reject_invalid_targets`
- `collision_lower_limits`
- `collision_upper_limits`
- `underground_limit`
- `num_collision_check_samples`
- `resample_max_retries`

新增逻辑：
- `_arm_lpy_to_local_xyz()`
- `_arm_target_collision_mask()`
- 在 `_resample_arm_commands()` 里，对采样到的 arm target 做轨迹插值检查
- 如果轨迹穿过碰撞盒，或 z 低于地下阈值，就重采样

效果：
- VR/键控时更不容易把末端目标设到机身内部或地下
- 训练和测试时的 arm 目标分布更安全、更合理

## 5. arm 目标改成带 delta curriculum 的连续采样

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`
- `go1_gym/envs/automatic/legged_robot_config.py`

新增配置：
- `max_delta_l`
- `max_delta_p`
- `max_delta_y`
- `max_delta_roll_ee`
- `max_delta_pitch_ee`
- `max_delta_yaw_ee`
- `delta_curriculum_start_iter`
- `delta_curriculum_end_iter`
- `delta_curriculum_power`

新增逻辑：
- `_apply_resample_delta_limit()`
- `_get_delta_curriculum_ratio()`
- `_get_curriculum_delta_limit()`

效果：
- arm target 不再每次完全随机跳变
- 目标变化更连续，更接近 VR 输入流或人工操作的连续性
- 训练后期再逐渐放开更大的目标变化范围

## 6. arm tracking reward 增加位置/姿态权重过渡

涉及文件：
- `go1_gym/envs/rewards/rewards.py`
- `go1_gym/envs/automatic/legged_robot_config.py`
- `scripts/auto_train.py`
- `scripts/unified_train.py`

新增配置：
- `manip_weight_lpy_start`
- `manip_weight_lpy_end`
- `manip_weight_rpy_start`
- `manip_weight_rpy_end`
- `manip_weight_keep_sum_constant`
- `manip_weight_transition_iters`
- `manip_weight_transition_power`

具体改动：
- `_reward_arm_manip_commands_tracking_combine()` 不再只使用固定的 `manip_weight_lpy / manip_weight_rpy`
- 现在会根据训练进度，在位置权重和姿态权重之间做过渡

效果：
- 可以先偏重位置跟踪，再逐步引入姿态约束
- 更适合把 VR 控制或远端目标的“先到位，再对姿态”分阶段学出来

## 7. 增加 reward 调试缓冲，便于在线观察 arm tracking 表现

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`

具体改动：
- 在 `compute_reward()` 中，单独保存：
  - `debug_arm_manip_commands_tracking_combine_raw`
  - `debug_arm_manip_commands_tracking_combine_scaled`

效果：
- play 脚本和调试脚本可以直接读取这条 reward
- 更方便判断问题来自控制、目标定义，还是 reward 几何不一致

## 8. 新增碰撞框可视化，直接看到 arm target 禁入区域

涉及文件：
- `go1_gym/envs/automatic/legged_robot.py`

具体改动：
- 新增 `_draw_arm_collision_bbox()`
- 在 viewer 中，以 base yaw 局部坐标系画出 arm command rejection box

效果：
- VR 联调时能直接看到“不安全目标区域”
- 更容易解释为什么某些目标被拒绝或重采样

## 9. 运行时参数从 `asset_config.py` 暴露出来，方便本地快速调 PD 和安全边界

涉及文件：
- `go1_gym/envs/go1/asset_config.py`
- `scripts/load_policy.py`

具体改动：
- 在 `asset_config.py` 里新增：
  - `ARM_STIFFNESS_ARM`
  - `ARM_DAMPING_ARM`
  - `ARM_COMMAND_SAFETY`
- 新增 helper：
  - `apply_arm_pd_from_asset_config()`
  - `apply_arm_sampling_safety_from_asset_config()`
- `load_env()` 支持：
  - `apply_asset_config_override`
  - `arm_kp`
  - `arm_kd`
  - `arm_joint_names`

效果：
- 不改训练存档，也能在本地 play/eval 时试不同 arm PD
- 可以快速切 arm 命令安全范围，适合 VR 联调

## 10. play / evaluate / draft 工具链增强，方便验证 VR 适配后的效果

涉及文件：
- `scripts/play_by_key.py`
- `scripts/play_by_key_arm_cycle.py`
- `scripts/evaluate_whole_body.py`
- `scripts/draft/search_arm_kp_kd.py`
- `scripts/draft/search_joint5_kpkd.py`
- `scripts/draft/test_arm_joint_tracking.py`
- `scripts/draft/test_arm_trajectory_tracking.py`

具体改动：
- `play_by_key.py` 增加 arm joint 实时曲线、tracking reward 曲线、collision 统计
- 支持运行时覆盖 arm `kp/kd`
- 新增 `play_by_key_arm_cycle.py` 用于固定轨迹验证
- 新增 `evaluate_whole_body.py`，可以分别评估：
  - 底盘不动只测 arm
  - arm 不动只测底盘
  - whole body 一起测

效果：
- 让 VR 适配后的联调不再只靠肉眼看 viewer
- 可以更系统地检查末端 tracking 和底盘 tracking

## 11. 训练入口的小改动

涉及文件：
- `scripts/auto_train.py`

具体改动：
- 支持 `--exp_name` 自定义训练目录末尾命名
- 本地日志路径不再自动追加 `_seedXX`
- `my_test_v3` 训练脚本中默认从你之前效果最好的 run 继续恢复

这部分不是 VR 核心逻辑，但和这次实验版本管理有关。

## 总结

如果只抓和“VR 适配”最相关的变化，`my_test_v3` 相比 `VR适配 可以通信` 的核心升级可以概括成 5 点：

1. EE 显示、EE 姿态、reward、末端 pose 计算统一到了同一套工具坐标系定义
2. 可视化时序调整后，marker 更能反映当前真实命令
3. arm target 采样加入安全约束，不容易给出碰撞或插地目标
4. arm target 改成连续变化，更接近 VR / 手动输入的使用方式
5. 本地调 PD、看 tracking、做 whole-body 评估的工具链更完整

如果后面还要继续做 VR 适配，最值得继续统一的地方是：
- 把 `scripts/vr_play/play_by_remote.py` 也接上同样的姿态偏置和调试显示
- 把“偏置后的 EE tip pose”再抽成一个统一 helper，避免 viewer / reward / eval 继续漂移

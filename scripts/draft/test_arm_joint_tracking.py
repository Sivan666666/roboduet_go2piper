import argparse
import csv
import os
from typing import List

import isaacgym

assert isaacgym
import torch

from scripts.load_policy import load_env


def parse_target(target_str: str, degrees: bool) -> torch.Tensor:
    values = [float(x.strip()) for x in target_str.split(",") if x.strip()]
    if len(values) != 6:
        raise ValueError(
            f"--target 需要 6 个数值，当前收到 {len(values)} 个: {target_str}"
        )
    t = torch.tensor(values, dtype=torch.float32)
    if degrees:
        t = torch.deg2rad(t)
    return t


def format_vec(v: torch.Tensor) -> str:
    return ", ".join([f"{x:.4f}" for x in v.tolist()])


def main():
    parser = argparse.ArgumentParser("Arm joint tracking test with current kp/kd")
    parser.add_argument("--logdir", type=str, required=True, help="训练 run 路径，需包含 parameters.pkl")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="6 个机械臂目标关节角，逗号分隔。默认单位 rad。例如: 0.0,0.8,-0.6,0.2,0.3,-0.2",
    )
    parser.add_argument("--degrees", action="store_true", help="将 --target 按 degree 解析")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=80)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--tol", type=float, default=0.05, help="到位阈值(rad)，默认 0.05")
    parser.add_argument("--csv_path", type=str, default="", help="可选：保存每步误差 CSV")
    args = parser.parse_args()

    target_joint = parse_target(args.target, args.degrees)

    # 延迟导入，避免项目包的循环导入触发
    from go1_gym.utils.global_switch import global_switch
    from go1_gym.envs.automatic import KeyboardWrapper, VelocityTrackingEasyEnv

    global_switch.open_switch()
    wrapper = VelocityTrackingEasyEnv if args.headless else KeyboardWrapper
    env, cfg = load_env(
        args.logdir,
        wrapper=wrapper,
        headless=args.headless,
        device=args.sim_device,
    )
    env.env.enable_viewer_sync = True
    env.reset()

    device = env.device
    arm_start = env.num_actions_loco
    arm_end = env.num_actions_loco + env.num_actions_arm
    arm_indices = torch.arange(arm_start, arm_end, device=device, dtype=torch.long)

    # 当前默认角 + action_scale => 目标策略动作
    default_arm = env.default_dof_pos[0, arm_indices].detach().clone()
    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)

    target_joint = target_joint.to(device)
    desired_action = (target_joint - default_arm) / action_scale
    desired_action_clipped = torch.clamp(desired_action, -clip_actions, clip_actions)

    if not torch.allclose(desired_action, desired_action_clipped):
        print(
            "[WARN] 部分关节动作超过 clip_actions，已被裁剪。"
            f"\nraw action: [{format_vec(desired_action.cpu())}]"
            f"\nclipped   : [{format_vec(desired_action_clipped.cpu())}]"
        )

    desired_action = desired_action_clipped

    print("=" * 90)
    print("Arm Joint Tracking Test")
    print(f"logdir: {args.logdir}")
    print(f"control_type: {cfg.control.control_type}, action_scale: {action_scale}, clip_actions: {clip_actions}")
    print(f"arm dof idx: {arm_start} ~ {arm_end - 1}")
    print("-" * 90)
    print("Arm DOF / kp / kd / limits:")
    for i, idx in enumerate(arm_indices.tolist()):
        name = env.dof_names[idx]
        kp = env.p_gains[idx].item()
        kd = env.d_gains[idx].item()
        low, high = env.dof_pos_limits[idx].tolist()
        print(f"  [{i}] {name:<14} kp={kp:7.3f} kd={kd:7.3f} limit=[{low: .3f}, {high: .3f}]")
    print("-" * 90)
    print(f"default_joint(rad): [{format_vec(default_arm.cpu())}]")
    print(f"target_joint (rad): [{format_vec(target_joint.cpu())}]")
    print(f"target_action     : [{format_vec(desired_action.cpu())}]")
    print("=" * 90)

    # 固定动作：狗腿置零，机械臂打目标
    dog_actions = torch.zeros((env.num_envs, env.num_actions_loco), dtype=torch.float, device=device)
    arm_actions = torch.zeros((env.num_envs, env.num_actions_arm), dtype=torch.float, device=device)
    arm_actions[:, :] = desired_action.unsqueeze(0)

    # 预热到静止附近
    for _ in range(args.warmup_steps):
        env.step(dog_actions, torch.zeros_like(arm_actions))

    rows: List[List[float]] = []
    settled_step = None
    peak_abs_err = torch.zeros(env.num_actions_arm, device=device)
    final_abs_err = None

    for step in range(1, args.steps + 1):
        env.step(dog_actions, arm_actions)
        q = env.dof_pos[0, arm_indices]
        err = target_joint - q
        abs_err = torch.abs(err)
        peak_abs_err = torch.maximum(peak_abs_err, abs_err)
        final_abs_err = abs_err

        if settled_step is None and torch.all(abs_err <= args.tol):
            settled_step = step

        if step % args.print_every == 0 or step == 1 or step == args.steps:
            print(
                f"step={step:4d} "
                f"q=[{format_vec(q.cpu())}] "
                f"abs_err=[{format_vec(abs_err.cpu())}] "
                f"mean_abs_err={abs_err.mean().item():.4f}"
            )

        if args.csv_path:
            row = [step]
            row.extend(q.detach().cpu().tolist())
            row.extend(err.detach().cpu().tolist())
            row.extend(abs_err.detach().cpu().tolist())
            rows.append(row)

    print("\n" + "=" * 90)
    print("Summary")
    print(f"tol(rad): {args.tol:.4f}")
    if settled_step is None:
        print("settled: NO (在给定步数内未全部进入阈值)")
    else:
        print(f"settled: YES, first_step={settled_step}")
    print(f"final_abs_err(rad): [{format_vec(final_abs_err.cpu())}]")
    print(f"peak_abs_err (rad): [{format_vec(peak_abs_err.cpu())}]")
    print(f"final_mean_abs_err(rad): {final_abs_err.mean().item():.4f}")
    print("=" * 90)

    if args.csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_path)), exist_ok=True)
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["step"]
            header += [f"q{i+1}" for i in range(6)]
            header += [f"err{i+1}" for i in range(6)]
            header += [f"abs_err{i+1}" for i in range(6)]
            writer.writerow(header)
            writer.writerows(rows)
        print(f"CSV saved: {args.csv_path}")


if __name__ == "__main__":
    main()

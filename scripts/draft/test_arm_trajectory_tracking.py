import argparse
import math

import isaacgym

assert isaacgym
import torch

from scripts.load_policy import load_env


def main():
    parser = argparse.ArgumentParser("Trajectory tracking test for arm joints")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--arm_kp", type=float, default=None, help="Override all arm joints kp")
    parser.add_argument("--arm_kd", type=float, default=None, help="Override all arm joints kd")
    parser.add_argument("--joint5_kp", type=float, default=None, help="Override piper_joint5 kp only")
    parser.add_argument("--joint5_kd", type=float, default=None, help="Override piper_joint5 kd only")
    args = parser.parse_args()

    # delayed import to avoid package-init circular issues
    from go1_gym.utils.global_switch import global_switch
    from go1_gym.envs.automatic import VelocityTrackingEasyEnv

    global_switch.open_switch()
    env, cfg = load_env(
        args.logdir,
        wrapper=VelocityTrackingEasyEnv,
        headless=args.headless,
        device=args.sim_device,
    )
    env.env.enable_viewer_sync = True
    # NOTE:
    # 当前工程里的 reset() 可能触发 _resample_arm_commands 中的 cpu/cuda 混用报错。
    # 这里不调用 reset，直接用 warmup 把状态拉到稳定区间。

    arm_start = env.num_actions_loco
    arm_end = env.num_actions_loco + env.num_actions_arm
    arm_idx = torch.arange(arm_start, arm_end, device=env.device, dtype=torch.long)
    arm_names = [env.dof_names[i] for i in arm_idx.tolist()]
    if "piper_joint5" not in arm_names:
        raise RuntimeError(f"piper_joint5 not found in arm dof names: {arm_names}")

    j5_local = arm_names.index("piper_joint5")
    j5_global = arm_idx[j5_local].item()

    # Optional runtime gain override.
    if args.arm_kp is not None:
        env.p_gains[arm_idx] = float(args.arm_kp)
    if args.arm_kd is not None:
        env.d_gains[arm_idx] = float(args.arm_kd)
    if args.joint5_kp is not None:
        env.p_gains[j5_global] = float(args.joint5_kp)
    if args.joint5_kd is not None:
        env.d_gains[j5_global] = float(args.joint5_kd)

    # In control_type='M', arm joints are effectively tracked by simulator DOF props
    # (stiffness/damping) with position targets. Update runtime actor props as well.
    for ei in range(env.num_envs):
        dof_props = env.gym.get_actor_dof_properties(env.envs[ei], env.actor_handles[ei])
        for local_i, global_i in enumerate(arm_idx.tolist()):
            dof_props["stiffness"][global_i] = float(env.p_gains[global_i].item())
            dof_props["damping"][global_i] = float(env.d_gains[global_i].item())
        env.gym.set_actor_dof_properties(env.envs[ei], env.actor_handles[ei], dof_props)

    act_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)
    default_arm = env.default_dof_pos[0, arm_idx].clone()
    limits = env.dof_pos_limits[arm_idx].clone()

    arm_zero = torch.zeros((env.num_envs, env.num_actions_arm), dtype=torch.float, device=env.device)
    dog_zero = torch.zeros((env.num_envs, env.num_actions_loco), dtype=torch.float, device=env.device)

    for _ in range(args.warmup_steps):
        env.step(dog_zero, arm_zero)

    err_abs_hist = []
    for k in range(args.steps):
        t = k * float(env.dt)

        target = default_arm.clone()
        # Smooth multi-joint trajectory
        target[0] = 0.15 * math.sin(2 * math.pi * 0.35 * t)
        target[1] = 0.60 + 0.22 * math.sin(2 * math.pi * 0.25 * t)
        target[2] = -0.50 + 0.25 * math.sin(2 * math.pi * 0.25 * t + 0.8)
        target[3] = 0.18 * math.sin(2 * math.pi * 0.30 * t + 0.3)
        target[4] = 0.75 * math.sin(2 * math.pi * 0.22 * t)
        target[5] = 0.90 * math.sin(2 * math.pi * 0.22 * t + 0.4)

        target = torch.max(torch.min(target, limits[:, 1]), limits[:, 0])
        action = torch.clamp((target - default_arm) / act_scale, -clip_actions, clip_actions)

        arm_cmd = torch.zeros_like(arm_zero)
        arm_cmd[:, :] = action.unsqueeze(0)
        env.step(dog_zero, arm_cmd)

        q = env.dof_pos[0, arm_idx]
        err_abs_hist.append(torch.abs(target - q))

    err_abs_hist = torch.stack(err_abs_hist, dim=0)  # [T,6]
    mae = err_abs_hist.mean(dim=0)
    rmse = torch.sqrt((err_abs_hist ** 2).mean(dim=0))
    p95 = torch.quantile(err_abs_hist, 0.95, dim=0)
    mx = err_abs_hist.max(dim=0).values

    print("=" * 90)
    print("Trajectory Tracking Report")
    print(f"logdir: {args.logdir}")
    print(
        "runtime arm gains: "
        f"kp={[round(float(env.p_gains[i].item()), 4) for i in arm_idx.tolist()]}, "
        f"kd={[round(float(env.d_gains[i].item()), 4) for i in arm_idx.tolist()]}"
    )
    print(f"steps={args.steps}, dt={float(env.dt):.4f}, action_scale={act_scale:.4f}")
    print("-" * 90)
    for i, name in enumerate(arm_names):
        print(
            f"{name:12s} MAE={mae[i].item():.4f} "
            f"RMSE={rmse[i].item():.4f} P95={p95[i].item():.4f} MAX={mx[i].item():.4f}"
        )
    print("-" * 90)
    print(f"All-joint MAE mean: {mae.mean().item():.4f} rad")
    print(f"Joint5 MAE        : {mae[j5_local].item():.4f} rad")
    print(f"Joint5 RMSE       : {rmse[j5_local].item():.4f} rad")
    print(f"Joint5 P95        : {p95[j5_local].item():.4f} rad")
    print(f"Joint5 MAX        : {mx[j5_local].item():.4f} rad")
    print("=" * 90)


if __name__ == "__main__":
    main()

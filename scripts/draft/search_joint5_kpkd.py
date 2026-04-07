import argparse
import itertools
import time
from typing import Dict, List

import isaacgym

assert isaacgym
import torch

from scripts.load_policy import load_env


def parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@torch.no_grad()
def evaluate_kp_kd(
    env,
    arm_indices: torch.Tensor,
    joint5_local_idx: int,
    joint5_global_idx: int,
    default_arm: torch.Tensor,
    target_joint5_vals: List[float],
    kp5: float,
    kd5: float,
    action_scale: float,
    clip_actions: float,
    warmup_steps: int,
    rollout_steps: int,
    eval_tail: int,
    dog_actions_zero: torch.Tensor,
) -> Dict[str, float]:
    env.p_gains[joint5_global_idx] = kp5
    env.d_gains[joint5_global_idx] = kd5

    j5_tail_mean_list = []
    all_tail_mean_list = []
    j5_peak = 0.0

    for j5_t in target_joint5_vals:
        env.reset()

        arm_zero = torch.zeros((env.num_envs, env.num_actions_arm), dtype=torch.float, device=env.device)
        for _ in range(warmup_steps):
            env.step(dog_actions_zero, arm_zero)

        target_arm = default_arm.clone()
        target_arm[joint5_local_idx] = j5_t
        action_arm = torch.clamp((target_arm - default_arm) / action_scale, -clip_actions, clip_actions)
        arm_cmd = torch.zeros_like(arm_zero)
        arm_cmd[:, :] = action_arm.unsqueeze(0)

        j5_err_hist = []
        all_err_hist = []
        for _ in range(rollout_steps):
            env.step(dog_actions_zero, arm_cmd)
            q = env.dof_pos[0, arm_indices]
            abs_err = torch.abs(target_arm - q)
            j5e = abs_err[joint5_local_idx].item()
            j5_err_hist.append(j5e)
            all_err_hist.append(abs_err.mean().item())
            if j5e > j5_peak:
                j5_peak = j5e

        tail = max(1, min(eval_tail, len(j5_err_hist)))
        j5_tail_mean_list.append(sum(j5_err_hist[-tail:]) / tail)
        all_tail_mean_list.append(sum(all_err_hist[-tail:]) / tail)

    return {
        "j5_mean_err": float(sum(j5_tail_mean_list) / len(j5_tail_mean_list)),
        "all_mean_err": float(sum(all_tail_mean_list) / len(all_tail_mean_list)),
        "j5_peak_err": float(j5_peak),
    }


def main():
    parser = argparse.ArgumentParser("Grid search for piper_joint5 kp/kd")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=45)
    parser.add_argument("--eval_tail", type=int, default=15)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--kp_list", type=str, default="5,8,10,12,15,20,25,30,40")
    parser.add_argument("--kd_list", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument(
        "--joint5_targets",
        type=str,
        default="-0.8,-0.4,0.0,0.4,0.8",
        help="joint5 目标角(rad)列表",
    )
    args = parser.parse_args()

    # delayed imports to avoid circular import at package init time
    from go1_gym.utils.global_switch import global_switch
    from go1_gym.envs.automatic import VelocityTrackingEasyEnv

    kp_list = parse_list(args.kp_list)
    kd_list = parse_list(args.kd_list)
    target_joint5_vals = parse_list(args.joint5_targets)

    global_switch.open_switch()
    env, cfg = load_env(
        args.logdir,
        wrapper=VelocityTrackingEasyEnv,
        headless=args.headless,
        device=args.sim_device,
    )
    env.env.enable_viewer_sync = True

    arm_start = env.num_actions_loco
    arm_end = env.num_actions_loco + env.num_actions_arm
    arm_indices = torch.arange(arm_start, arm_end, device=env.device, dtype=torch.long)
    arm_names = [env.dof_names[i] for i in arm_indices.tolist()]

    if "piper_joint5" not in arm_names:
        raise RuntimeError(f"未在 arm dof 中找到 piper_joint5, arm_names={arm_names}")
    joint5_local_idx = arm_names.index("piper_joint5")
    joint5_global_idx = arm_indices[joint5_local_idx].item()

    base_kp = env.p_gains[arm_indices].detach().clone()
    base_kd = env.d_gains[arm_indices].detach().clone()
    default_arm = env.default_dof_pos[0, arm_indices].detach().clone()

    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)
    dog_actions_zero = torch.zeros((env.num_envs, env.num_actions_loco), dtype=torch.float, device=env.device)

    print("=" * 100)
    print("Joint5 KP/KD Search Start")
    print(f"logdir: {args.logdir}")
    print(f"joint5 global idx: {joint5_global_idx}, local idx in arm: {joint5_local_idx}")
    print(f"base joint5 kp={base_kp[joint5_local_idx].item():.4f}, kd={base_kd[joint5_local_idx].item():.4f}")
    print(f"grid size: {len(kp_list)} x {len(kd_list)} = {len(kp_list) * len(kd_list)}")
    print(f"joint5 targets(rad): {target_joint5_vals}")
    print("=" * 100)

    results = []
    st = time.time()

    all_pairs = list(itertools.product(kp_list, kd_list))
    for i, (kp5, kd5) in enumerate(all_pairs, start=1):
        m = evaluate_kp_kd(
            env=env,
            arm_indices=arm_indices,
            joint5_local_idx=joint5_local_idx,
            joint5_global_idx=joint5_global_idx,
            default_arm=default_arm,
            target_joint5_vals=target_joint5_vals,
            kp5=kp5,
            kd5=kd5,
            action_scale=action_scale,
            clip_actions=clip_actions,
            warmup_steps=args.warmup_steps,
            rollout_steps=args.rollout_steps,
            eval_tail=args.eval_tail,
            dog_actions_zero=dog_actions_zero,
        )
        row = {"kp5": kp5, "kd5": kd5, **m}
        results.append(row)

        if i % 6 == 0 or i == 1 or i == len(all_pairs):
            best_now = min(results, key=lambda x: (x["j5_mean_err"], x["all_mean_err"]))
            print(
                f"[{i:3d}/{len(all_pairs)}] "
                f"curr(j5={m['j5_mean_err']:.5f}, all={m['all_mean_err']:.5f}) "
                f"best(j5={best_now['j5_mean_err']:.5f}, all={best_now['all_mean_err']:.5f}) "
                f"elapsed={time.time() - st:.1f}s"
            )

    results_sorted = sorted(results, key=lambda x: (x["j5_mean_err"], x["all_mean_err"]))
    best = results_sorted[0]

    print("\n" + "=" * 100)
    print("Top Results (sorted by joint5 mean abs error)")
    for rank, r in enumerate(results_sorted[: args.topk], start=1):
        print(
            f"{rank:2d}. kp5={r['kp5']:7.3f}, kd5={r['kd5']:6.3f}, "
            f"j5_mean_err={r['j5_mean_err']:.6f}, all_mean_err={r['all_mean_err']:.6f}, j5_peak={r['j5_peak_err']:.6f}"
        )

    print("\nBest:")
    print(f"  piper_joint5 kp={best['kp5']:.6f}, kd={best['kd5']:.6f}")
    print(
        f"  metrics: j5_mean_err={best['j5_mean_err']:.6f}, "
        f"all_mean_err={best['all_mean_err']:.6f}, j5_peak={best['j5_peak_err']:.6f}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

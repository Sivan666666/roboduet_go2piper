import argparse
import itertools
import math
import time
from typing import Dict, List, Tuple

import isaacgym

assert isaacgym
import torch

from scripts.load_policy import load_env


def parse_targets(targets_str: str, degrees: bool) -> List[torch.Tensor]:
    """
    Format:
      "0,0.6,-0.5,0,0,0;0,0.6,-0.5,0,0.5,0.5"
    """
    targets = []
    for seg in targets_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        vals = [float(x.strip()) for x in seg.split(",") if x.strip()]
        if len(vals) != 6:
            raise ValueError(f"每个 target 需要 6 个值，当前: {seg}")
        t = torch.tensor(vals, dtype=torch.float32)
        if degrees:
            t = torch.deg2rad(t)
        targets.append(t)
    if len(targets) == 0:
        raise ValueError("未解析到任何 target")
    return targets


def parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_candidates(args) -> List[Tuple[float, float, float, float]]:
    # stage1 coarse
    coarse = list(
        itertools.product(
            args.kp_scales,
            args.kd_scales,
            args.wrist_kp_scales,
            args.wrist_kd_scales,
        )
    )
    if not args.refine:
        return coarse

    # stage2 refine around best coarse later in runtime
    return coarse


def to_action(
    target_joint: torch.Tensor,
    default_arm: torch.Tensor,
    action_scale: float,
    clip_actions: float,
) -> torch.Tensor:
    a = (target_joint - default_arm) / action_scale
    return torch.clamp(a, -clip_actions, clip_actions)


@torch.no_grad()
def eval_candidate(
    env,
    arm_indices: torch.Tensor,
    default_arm: torch.Tensor,
    action_scale: float,
    clip_actions: float,
    dog_actions_zero: torch.Tensor,
    kp_candidate: torch.Tensor,
    kd_candidate: torch.Tensor,
    targets: List[torch.Tensor],
    warmup_steps: int,
    rollout_steps: int,
    eval_tail: int,
) -> Dict[str, float]:
    # apply gains
    env.p_gains[arm_indices] = kp_candidate
    env.d_gains[arm_indices] = kd_candidate

    per_target_mean = []
    per_target_wrist_mean = []
    per_target_max = []

    for target in targets:
        env.reset()

        # warmup with zero action near default pose
        arm_zero = torch.zeros((env.num_envs, env.num_actions_arm), dtype=torch.float, device=env.device)
        for _ in range(warmup_steps):
            env.step(dog_actions_zero, arm_zero)

        arm_action = torch.zeros_like(arm_zero)
        arm_action[:, :] = to_action(
            target.to(env.device), default_arm, action_scale, clip_actions
        ).unsqueeze(0)

        err_hist = []
        wrist_err_hist = []
        max_err = 0.0

        for _ in range(rollout_steps):
            env.step(dog_actions_zero, arm_action)
            q = env.dof_pos[0, arm_indices]
            abs_err = torch.abs(target.to(env.device) - q)
            err_hist.append(abs_err.mean().item())
            wrist_err_hist.append(abs_err[4:6].mean().item())
            max_err = max(max_err, abs_err.max().item())

        tail = max(1, min(eval_tail, len(err_hist)))
        mean_err = float(sum(err_hist[-tail:]) / tail)
        mean_wrist_err = float(sum(wrist_err_hist[-tail:]) / tail)

        per_target_mean.append(mean_err)
        per_target_wrist_mean.append(mean_wrist_err)
        per_target_max.append(max_err)

    return {
        "mean_err": float(sum(per_target_mean) / len(per_target_mean)),
        "mean_wrist_err": float(sum(per_target_wrist_mean) / len(per_target_wrist_mean)),
        "max_err": float(max(per_target_max)),
    }


def refine_around_best(best, base_grid_scale=0.15):
    # Multiply best scales by local factors
    local = [1.0 - base_grid_scale, 1.0, 1.0 + base_grid_scale]
    cands = []
    for a, b, c, d in itertools.product(local, local, local, local):
        cands.append(
            (
                max(0.1, best[0] * a),
                max(0.1, best[1] * b),
                max(0.1, best[2] * c),
                max(0.1, best[3] * d),
            )
        )
    # unique with rounded key
    uniq = {}
    for x in cands:
        key = tuple([round(v, 6) for v in x])
        uniq[key] = x
    return list(uniq.values())


def main():
    parser = argparse.ArgumentParser("Search kp/kd with minimum arm tracking error")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--degrees", action="store_true")
    parser.add_argument(
        "--targets",
        type=str,
        default="0,0.6,-0.5,0,0,0;0,0.6,-0.5,0,0.5,0.5;0,0.6,-0.5,0,-0.5,-0.5;0.2,0.9,-0.9,0.3,0.6,-0.6",
        help="多个目标关节角，用 ; 分隔，每个目标6维",
    )
    parser.add_argument("--warmup_steps", type=int, default=15)
    parser.add_argument("--rollout_steps", type=int, default=90)
    parser.add_argument("--eval_tail", type=int, default=30)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--refine", action="store_true", default=True)

    parser.add_argument("--kp_scales", type=str, default="0.8,1.0,1.2,1.5")
    parser.add_argument("--kd_scales", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--wrist_kp_scales", type=str, default="1.0,1.5,2.0,3.0")
    parser.add_argument("--wrist_kd_scales", type=str, default="1.0,1.5")

    args = parser.parse_args()

    # delayed imports to avoid package circular init issue
    from go1_gym.utils.global_switch import global_switch
    from go1_gym.envs.automatic import VelocityTrackingEasyEnv

    args.kp_scales = parse_list(args.kp_scales)
    args.kd_scales = parse_list(args.kd_scales)
    args.wrist_kp_scales = parse_list(args.wrist_kp_scales)
    args.wrist_kd_scales = parse_list(args.wrist_kd_scales)
    targets = parse_targets(args.targets, args.degrees)

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

    base_kp = env.p_gains[arm_indices].detach().clone()
    base_kd = env.d_gains[arm_indices].detach().clone()
    default_arm = env.default_dof_pos[0, arm_indices].detach().clone()
    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)
    dog_actions_zero = torch.zeros((env.num_envs, env.num_actions_loco), dtype=torch.float, device=env.device)

    print("=" * 100)
    print("KP/KD Search Start")
    print(f"logdir: {args.logdir}")
    print(f"targets: {len(targets)}")
    print(f"base_kp: {[round(x,4) for x in base_kp.tolist()]}")
    print(f"base_kd: {[round(x,4) for x in base_kd.tolist()]}")
    print(
        f"grid coarse size: {len(args.kp_scales) * len(args.kd_scales) * len(args.wrist_kp_scales) * len(args.wrist_kd_scales)}"
    )
    print("=" * 100)

    coarse_candidates = build_candidates(args)
    results = []
    st = time.time()

    for idx, (kp_s, kd_s, wk_s, wd_s) in enumerate(coarse_candidates, start=1):
        kp_c = base_kp * kp_s
        kd_c = base_kd * kd_s
        kp_c[4:6] = kp_c[4:6] * wk_s
        kd_c[4:6] = kd_c[4:6] * wd_s

        met = eval_candidate(
            env=env,
            arm_indices=arm_indices,
            default_arm=default_arm,
            action_scale=action_scale,
            clip_actions=clip_actions,
            dog_actions_zero=dog_actions_zero,
            kp_candidate=kp_c,
            kd_candidate=kd_c,
            targets=targets,
            warmup_steps=args.warmup_steps,
            rollout_steps=args.rollout_steps,
            eval_tail=args.eval_tail,
        )
        row = {
            "kp_scale": kp_s,
            "kd_scale": kd_s,
            "wrist_kp_scale": wk_s,
            "wrist_kd_scale": wd_s,
            **met,
        }
        results.append(row)

        if idx % 8 == 0 or idx == 1 or idx == len(coarse_candidates):
            elapsed = time.time() - st
            best_now = min(results, key=lambda r: r["mean_err"])
            print(
                f"[coarse {idx:3d}/{len(coarse_candidates)}] "
                f"curr_mean={met['mean_err']:.5f}, curr_wrist={met['mean_wrist_err']:.5f}, "
                f"best_mean={best_now['mean_err']:.5f}, elapsed={elapsed:.1f}s"
            )

    best_coarse = min(results, key=lambda r: r["mean_err"])

    # optional refine
    if args.refine:
        print("\nRefine around best coarse...")
        ref_list = refine_around_best(
            (
                best_coarse["kp_scale"],
                best_coarse["kd_scale"],
                best_coarse["wrist_kp_scale"],
                best_coarse["wrist_kd_scale"],
            ),
            base_grid_scale=0.15,
        )
        ref_results = []
        for idx, (kp_s, kd_s, wk_s, wd_s) in enumerate(ref_list, start=1):
            kp_c = base_kp * kp_s
            kd_c = base_kd * kd_s
            kp_c[4:6] = kp_c[4:6] * wk_s
            kd_c[4:6] = kd_c[4:6] * wd_s

            met = eval_candidate(
                env=env,
                arm_indices=arm_indices,
                default_arm=default_arm,
                action_scale=action_scale,
                clip_actions=clip_actions,
                dog_actions_zero=dog_actions_zero,
                kp_candidate=kp_c,
                kd_candidate=kd_c,
                targets=targets,
                warmup_steps=args.warmup_steps,
                rollout_steps=args.rollout_steps,
                eval_tail=args.eval_tail,
            )
            ref_results.append(
                {
                    "kp_scale": kp_s,
                    "kd_scale": kd_s,
                    "wrist_kp_scale": wk_s,
                    "wrist_kd_scale": wd_s,
                    **met,
                }
            )

            if idx % 10 == 0 or idx == 1 or idx == len(ref_list):
                print(f"[refine {idx:3d}/{len(ref_list)}] curr_mean={met['mean_err']:.5f}")

        results.extend(ref_results)

    results_sorted = sorted(results, key=lambda r: r["mean_err"])
    best = results_sorted[0]

    # recover actual kp/kd vectors
    best_kp = base_kp * best["kp_scale"]
    best_kd = base_kd * best["kd_scale"]
    best_kp[4:6] = best_kp[4:6] * best["wrist_kp_scale"]
    best_kd[4:6] = best_kd[4:6] * best["wrist_kd_scale"]

    print("\n" + "=" * 100)
    print("Top Results")
    for i, r in enumerate(results_sorted[: args.topk], start=1):
        print(
            f"{i:2d}. mean_err={r['mean_err']:.6f}, wrist={r['mean_wrist_err']:.6f}, max={r['max_err']:.6f}, "
            f"kp_s={r['kp_scale']:.4f}, kd_s={r['kd_scale']:.4f}, wk_s={r['wrist_kp_scale']:.4f}, wd_s={r['wrist_kd_scale']:.4f}"
        )

    print("\nBest KP/KD (apply to piper_joint1~6):")
    names = [env.dof_names[i] for i in arm_indices.tolist()]
    for i, n in enumerate(names):
        print(f"  {n}: kp={best_kp[i].item():.6f}, kd={best_kd[i].item():.6f}")

    print(
        f"\nBest metric: mean_err={best['mean_err']:.6f}, "
        f"mean_wrist_err={best['mean_wrist_err']:.6f}, max_err={best['max_err']:.6f}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

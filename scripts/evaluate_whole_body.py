import argparse
import json
import os
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import isaacgym

assert isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate

from go1_gym.utils.common import quaternion_to_rpy
from go1_gym.utils.math_utils import wrap_to_pi
from scripts.load_policy import load_arm_policy, load_dog_policy, load_env


SCENARIO_ALL = ("arm_only", "base_only", "whole_body")


@dataclass
class TrialCommand:
    dog_cmd: List[float]
    arm_lpy: List[float]
    arm_rpy: List[float]


def set_random_seed(seed: int) -> int:
    if seed < 0:
        seed = random.randint(0, 100000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def tensor_to_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().view(-1).tolist()]


def summarize_tensor_mean_abs(raw_values: List[torch.Tensor]) -> Dict[str, List[float]]:
    if not raw_values:
        return {}
    values = torch.stack([v.detach().cpu().float() for v in raw_values], dim=0)
    abs_values = torch.abs(values)
    return {"mean_abs": tensor_to_list(abs_values.mean(dim=0))}


def summarize_scalar_mean_abs(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    x = torch.tensor(values, dtype=torch.float32)
    return {"mean_abs": float(torch.abs(x).mean().item())}


def seconds_to_steps(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / float(dt))))


def build_default_results_path(logdir: str, ckptid: str, scenario: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    run_leaf = Path(logdir).name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{run_leaf}_ckpt{ckptid}_{scenario}_{timestamp}.json"
    return str(results_dir / filename)


def center_of_range(low: float, high: float) -> float:
    return 0.5 * (float(low) + float(high))


def sample_dog_command(cfg, device: torch.device) -> torch.Tensor:
    for _ in range(100):
        cmd = torch.tensor(
            [
                random.uniform(*cfg.commands.lin_vel_x),
                random.uniform(*cfg.commands.lin_vel_y),
                random.uniform(*cfg.commands.ang_vel_yaw),
            ],
            dtype=torch.float32,
            device=device,
        )
        if torch.norm(cmd[:2]).item() > 0.1 or abs(float(cmd[2].item())) > 0.1:
            return cmd
    return torch.tensor(
        [
            center_of_range(*cfg.commands.lin_vel_x),
            center_of_range(*cfg.commands.lin_vel_y),
            center_of_range(*cfg.commands.ang_vel_yaw),
        ],
        dtype=torch.float32,
        device=device,
    )


def sample_arm_target(env) -> Dict[str, torch.Tensor]:
    device = env.device
    cfg = env.cfg.arm.commands
    start_lpy = env.commands_arm[0:1, 0:3].clone()
    reject_invalid = bool(getattr(cfg, "reject_invalid_targets", True))
    max_retries = max(1, int(getattr(cfg, "resample_max_retries", 10)))

    for _ in range(max_retries):
        lpy = torch.tensor(
            [
                random.uniform(*cfg.l),
                random.uniform(*cfg.p),
                random.uniform(*cfg.y),
            ],
            dtype=torch.float32,
            device=device,
        ).view(1, 3)
        if reject_invalid:
            invalid = env._arm_target_collision_mask(start_lpy, lpy)
            if bool(invalid[0].item()):
                continue
        break
    else:
        lpy = torch.tensor(
            [
                center_of_range(*cfg.l),
                center_of_range(*cfg.p),
                center_of_range(*cfg.y),
            ],
            dtype=torch.float32,
            device=device,
        ).view(1, 3)

    rpy = torch.tensor(
        [
            random.uniform(*cfg.roll_ee),
            random.uniform(*cfg.pitch_ee),
            random.uniform(*cfg.yaw_ee),
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 3)

    return {"lpy": lpy, "rpy": rpy}


def get_neutral_arm_target(env) -> Dict[str, torch.Tensor]:
    cfg = env.cfg.arm.commands
    device = env.device
    lpy = torch.tensor(
        [
            center_of_range(*cfg.l),
            center_of_range(*cfg.p),
            center_of_range(*cfg.y),
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 3)
    rpy = torch.tensor(
        [
            center_of_range(*cfg.roll_ee),
            center_of_range(*cfg.pitch_ee),
            center_of_range(*cfg.yaw_ee),
        ],
        dtype=torch.float32,
        device=device,
    ).view(1, 3)
    return {"lpy": lpy, "rpy": rpy}


def set_dog_command(env, dog_cmd: torch.Tensor) -> None:
    env.commands_dog[:, 0] = dog_cmd[0]
    env.commands_dog[:, 1] = dog_cmd[1]
    env.commands_dog[:, 2] = dog_cmd[2]
    env.commands_dog[:, 3] = 0.0
    env.commands_dog[:, 4] = 0.0


def set_arm_target(env, lpy: torch.Tensor, rpy: torch.Tensor) -> None:
    env.commands_arm[:, 0:3] = lpy
    env.commands_arm[:, 3:6] = rpy
    env.commands_arm_obs[:, 0:3] = lpy

    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]
    zero = torch.zeros_like(roll)
    q1 = quat_from_euler_xyz(zero, zero, yaw)
    q2 = quat_from_euler_xyz(zero, pitch, zero)
    q3 = quat_from_euler_xyz(roll, zero, zero)
    quats = quat_mul(q1, quat_mul(q2, q3))

    env.obj_quats[:] = quats
    env.visual_rpy[:] = quaternion_to_rpy(quats).to(env.device)
    target_abg = env.quat_to_angle(quats)
    env.target_abg[:] = target_abg
    env.commands_arm_obs[:, 3:6] = target_abg


def compute_arm_metrics(env) -> Dict[str, torch.Tensor]:
    env_ids = torch.arange(env.num_envs, device=env.device)
    ee_quat_raw = env.end_effector_state[env_ids, 3:7]
    ee_quat_yellow = quat_mul(ee_quat_raw, env.ee_rot_offset[env_ids])

    forward = quat_apply(env.base_quat[env_ids], env.forward_vec[env_ids])
    yaw_base = torch.atan2(forward[:, 1], forward[:, 0])

    grasper_move = torch.tensor([0.12, 0.0, 0.0], device=env.device, dtype=torch.float32).repeat((len(env_ids), 1))
    grasper_world = env.end_effector_state[env_ids, :3] + quat_rotate(ee_quat_yellow, grasper_move)

    x = torch.cos(yaw_base) * (grasper_world[:, 0] - env.root_states[env_ids, 0]) + torch.sin(yaw_base) * (
        grasper_world[:, 1] - env.root_states[env_ids, 1]
    )
    y = -torch.sin(yaw_base) * (grasper_world[:, 0] - env.root_states[env_ids, 0]) + torch.cos(yaw_base) * (
        grasper_world[:, 1] - env.root_states[env_ids, 1]
    )
    z = torch.mean(grasper_world[:, 2].unsqueeze(1) - env.measured_heights, dim=1) - 0.38

    l = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    p = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    yaw_tip = torch.atan2(y, x)
    lpy_actual = torch.stack([l, p, yaw_tip], dim=-1)
    lpy_delta = lpy_actual - env.commands_arm_obs[:, 0:3]
    roll_cmd = env.visual_rpy[:, 0]
    pitch_cmd = env.visual_rpy[:, 1]
    yaw_cmd = env.visual_rpy[:, 2]
    quat_cmd_base = quat_from_euler_xyz(roll_cmd, pitch_cmd, yaw_cmd)
    quat_cmd_world = quat_mul(env.base_quat, quat_cmd_base)
    rpy_target_vis = env.quat_to_angle(quat_cmd_world)

    rpy_actual_vis = env.quat_to_angle(ee_quat_yellow)
    rpy_delta = wrap_to_pi(rpy_actual_vis - rpy_target_vis)
    return {
        "lpy_delta": lpy_delta[0],
        "rpy_delta": rpy_delta[0],
    }


def compute_base_metrics(env) -> Dict[str, torch.Tensor]:
    lin_delta = env.base_lin_vel[0, :2] - env.commands_dog[0, :2]
    yaw_delta = env.base_ang_vel[0, 2] - env.commands_dog[0, 2]
    return {
        "lin_vel_delta": lin_delta,
        "yaw_vel_delta": yaw_delta,
    }


def get_timeout_flag(info: Dict) -> bool:
    time_outs = info.get("time_outs")
    if time_outs is None:
        return False
    if isinstance(time_outs, torch.Tensor):
        return bool(time_outs[0].detach().cpu().item())
    if isinstance(time_outs, (list, tuple)):
        return bool(time_outs[0])
    return bool(time_outs)


def step_policies(env, dog_policy, arm_policy, clip_actions: float, enable_plan: bool) -> Dict:
    with torch.no_grad():
        arm_obs = env.get_arm_observations()
        actions_arm_full = arm_policy(arm_obs)
        if enable_plan:
            env.plan(actions_arm_full[..., -2:])
        arm_action = torch.clamp(actions_arm_full[..., :-2], -clip_actions, clip_actions)

        dog_obs = env.get_dog_observations()
        actions_dog = dog_policy(dog_obs)
        _, _, done, info = env.step(actions_dog, arm_action)

    return {"done": bool(done[0].detach().cpu().item()), "info": info}


def prepare_trial(env, scenario: str) -> TrialCommand:
    device = env.device
    zero_dog = torch.zeros(3, dtype=torch.float32, device=device)
    neutral_arm = get_neutral_arm_target(env)

    if scenario == "arm_only":
        dog_cmd = zero_dog
        arm_target = sample_arm_target(env)
    elif scenario == "base_only":
        dog_cmd = sample_dog_command(env.cfg, device)
        arm_target = neutral_arm
    elif scenario == "whole_body":
        dog_cmd = sample_dog_command(env.cfg, device)
        arm_target = sample_arm_target(env)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    set_dog_command(env, dog_cmd)
    set_arm_target(env, arm_target["lpy"], arm_target["rpy"])
    return TrialCommand(
        dog_cmd=tensor_to_list(dog_cmd),
        arm_lpy=tensor_to_list(arm_target["lpy"][0]),
        arm_rpy=tensor_to_list(arm_target["rpy"][0]),
    )


def run_scenario(
    env,
    dog_policy,
    arm_policy,
    scenario: str,
    num_trials: int,
    settle_steps: int,
    metric_steps: int,
    clip_actions: float,
) -> Dict:
    print(f"\n{'=' * 100}\nScenario: {scenario}\n{'=' * 100}")

    arm_lpy_errors: List[torch.Tensor] = []
    arm_rpy_errors: List[torch.Tensor] = []
    base_lin_errors: List[torch.Tensor] = []
    base_yaw_errors: List[torch.Tensor] = []
    successful_trials = 0
    total_alive_steps = 0
    total_requested_steps = num_trials * (settle_steps + metric_steps)
    total_eval_steps = 0
    reset_count = 0

    for trial_idx in range(num_trials):
        env.reset()
        command = prepare_trial(env, scenario)
        enable_plan = scenario == "whole_body"

        trial_alive_steps = 0
        trial_eval_steps = 0
        done = False
        timeout = False

        for step_idx in range(settle_steps + metric_steps):
            step_out = step_policies(
                env,
                dog_policy=dog_policy,
                arm_policy=arm_policy,
                clip_actions=clip_actions,
                enable_plan=enable_plan,
            )
            done = step_out["done"]
            timeout = get_timeout_flag(step_out["info"])
            trial_alive_steps += 1
            total_alive_steps += 1

            if done:
                reset_count += 1
                break

            if step_idx >= settle_steps:
                arm_metrics = compute_arm_metrics(env)
                base_metrics = compute_base_metrics(env)

                arm_lpy_errors.append(arm_metrics["lpy_delta"])
                arm_rpy_errors.append(arm_metrics["rpy_delta"])
                base_lin_errors.append(base_metrics["lin_vel_delta"])
                base_yaw_errors.append(base_metrics["yaw_vel_delta"].view(1))

                trial_eval_steps += 1
                total_eval_steps += 1

        success = (not done) and trial_eval_steps == metric_steps
        if success:
            successful_trials += 1

        print(
            f"[{scenario}] trial={trial_idx + 1:03d}/{num_trials} "
            f"success={success} reset={done} timeout={timeout} "
            f"metric_steps={trial_eval_steps}/{metric_steps} "
            f"dog_cmd={np.round(np.array(command.dog_cmd), 3).tolist()} "
            f"arm_lpy={np.round(np.array(command.arm_lpy), 3).tolist()} "
            f"arm_rpy={np.round(np.array(command.arm_rpy), 3).tolist()}"
        )

    scenario_result = {
        "scenario": scenario,
        "num_trials": num_trials,
        "settle_steps": settle_steps,
        "metric_steps": metric_steps,
        "trial_survival_rate": successful_trials / max(1, num_trials),
        "step_survival_rate": total_alive_steps / max(1, total_requested_steps),
        "metric_coverage_rate": total_eval_steps / max(1, num_trials * metric_steps),
        "reset_count": reset_count,
        "arm_tracking": {
            "lpy_error": summarize_tensor_mean_abs(arm_lpy_errors),
            "rpy_error": summarize_tensor_mean_abs(arm_rpy_errors),
        },
        "base_tracking": {
            "lin_vel_error_xy": summarize_tensor_mean_abs(base_lin_errors),
            "yaw_vel_error": summarize_tensor_mean_abs(base_yaw_errors),
        },
    }
    return scenario_result


def print_summary(result: Dict) -> None:
    print(f"\nScenario `{result['scenario']}`")
    print(
        f"  trial_survival_rate={result['trial_survival_rate']:.4f}, "
        f"step_survival_rate={result['step_survival_rate']:.4f}, "
        f"metric_coverage_rate={result['metric_coverage_rate']:.4f}, "
        f"reset_count={result['reset_count']}"
    )

    arm = result["arm_tracking"]
    base = result["base_tracking"]
    if arm["lpy_error"]:
        print(
            "  arm lpy_error mean_abs="
            f"{np.round(np.array(arm['lpy_error']['mean_abs']), 4).tolist()}, "
            f"rpy_error mean_abs={np.round(np.array(arm['rpy_error']['mean_abs']), 4).tolist()}"
        )
    if base["lin_vel_error_xy"]:
        print(
            "  base lin_vel_error_xy mean_abs="
            f"{np.round(np.array(base['lin_vel_error_xy']['mean_abs']), 4).tolist()}, "
            f"yaw_vel_error mean_abs={np.round(np.array(base['yaw_vel_error']['mean_abs']), 4).tolist()}"
        )


def main():
    parser = argparse.ArgumentParser("Evaluate whole-body tracking on a trained run")
    parser.add_argument("--logdir", type=str, required=True, help="训练 run 路径，需包含 parameters.pkl")
    parser.add_argument("--ckptid", type=str, default="last", help="checkpoint id，例如 40000 或 last")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--settle_seconds", type=float, default=4.0, help="每个 target 先给多少秒收敛")
    parser.add_argument("--metric_seconds", type=float, default=2.0, help="随后多少秒用于统计平均误差")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "arm_only", "base_only", "whole_body"],
        help="选择评测场景",
    )
    parser.add_argument("--json_out", type=str, default="", help="可选：自定义 JSON 输出路径；默认保存到 results/")
    args = parser.parse_args()

    args.seed = set_random_seed(args.seed)

    from go1_gym.envs.automatic import VelocityTrackingEasyEnv
    from go1_gym.utils.global_switch import global_switch

    global_switch.open_switch()

    env, cfg = load_env(
        args.logdir,
        wrapper=VelocityTrackingEasyEnv,
        headless=args.headless,
        device=args.sim_device,
    )
    env.env.enable_viewer_sync = True
    dog_policy = load_dog_policy(args.logdir, args.ckptid, cfg)
    arm_policy = load_arm_policy(args.logdir, args.ckptid, cfg)

    clip_actions = float(cfg.normalization.clip_actions)
    settle_steps = seconds_to_steps(args.settle_seconds, env.dt)
    metric_steps = seconds_to_steps(args.metric_seconds, env.dt)
    scenarios = SCENARIO_ALL if args.scenario == "all" else (args.scenario,)

    print("=" * 100)
    print("Whole-Body Evaluation")
    print(f"logdir     : {args.logdir}")
    print(f"checkpoint : {args.ckptid}")
    print(f"sim_device : {args.sim_device}")
    print(f"headless   : {args.headless}")
    print(f"seed       : {args.seed}")
    print(f"num_trials : {args.num_trials}")
    print(f"dt         : {float(env.dt):.6f}")
    print(f"settle     : {args.settle_seconds:.2f}s ({settle_steps} steps)")
    print(f"metric     : {args.metric_seconds:.2f}s ({metric_steps} steps)")
    print(f"scenarios  : {list(scenarios)}")
    print("=" * 100)

    results = {
        "logdir": args.logdir,
        "checkpoint": args.ckptid,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "dt": float(env.dt),
        "settle_seconds": args.settle_seconds,
        "settle_steps": settle_steps,
        "metric_seconds": args.metric_seconds,
        "metric_steps": metric_steps,
        "scenarios": {},
    }

    for scenario in scenarios:
        scenario_result = run_scenario(
            env,
            dog_policy=dog_policy,
            arm_policy=arm_policy,
            scenario=scenario,
            num_trials=args.num_trials,
            settle_steps=settle_steps,
            metric_steps=metric_steps,
            clip_actions=clip_actions,
        )
        results["scenarios"][scenario] = scenario_result
        print_summary(scenario_result)

    json_out = args.json_out or build_default_results_path(
        logdir=args.logdir,
        ckptid=args.ckptid,
        scenario=args.scenario,
    )
    json_dir = os.path.dirname(os.path.abspath(json_out))
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to: {json_out}")


if __name__ == "__main__":
    main()

import argparse
import time

import isaacgym

assert isaacgym
import torch

from go1_gym.envs import *
from go1_gym.envs.automatic import KeyboardWrapper
from scripts.load_policy import load_dog_policy, load_env

logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0

# Small-amplitude waypoints in joint-angle space (rad), arm 6-DoF.
ARM_WAYPOINTS_SMALL_RAD = [
    [0.00, 0.60, -0.50, 0.00, 0.00, 0.00],
    [0.15, 0.78, -0.72, 0.16, 0.35, -0.28],
    [-0.15, 0.74, -0.68, -0.16, -0.35, 0.28],
    [0.10, 0.68, -0.60, 0.12, 0.22, -0.18],
]

# Medium/large-amplitude waypoints (still conservative, not extreme).
ARM_WAYPOINTS_LARGE_RAD = [
    [0.00, 0.60, -0.50, 0.00, 0.00, 0.00],
    [0.42, 1.10, -1.18, 0.50, 0.85, -0.75],
    [-0.42, 1.12, -1.22, -0.50, -0.85, 0.75],
    [0.32, 0.95, -0.96, 0.36, 0.62, -0.55],
]

SEGMENT_STEPS = 120
NUM_EVAL_STEPS = 30000

ENABLE_ARM_JOINT_PLOT = True
PLOT_NUM_JOINTS = 6
PLOT_HISTORY_S = 12.0
PLOT_UPDATE_EVERY = 2

PRINT_TRACKING_ERR_INTERVAL_S = 5.0


class ArmTrackingPlotter:
    def __init__(self, joint_names, history_s=10.0, update_every=2):
        self.joint_names = joint_names
        self.history_s = float(history_s)
        self.update_every = max(1, int(update_every))
        self.enabled = False

        self.t_hist = []
        self.target_hist = [[] for _ in joint_names]
        self.actual_hist = [[] for _ in joint_names]
        self.reward_hist = []

        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            self.plt.ion()
            self.fig, (self.ax_joint, self.ax_reward) = self.plt.subplots(
                2,
                1,
                figsize=(11, 7.0),
                sharex=True,
                gridspec_kw={"height_ratios": [3.0, 1.2]},
            )
            self.target_lines = []
            self.actual_lines = []
            for name in joint_names:
                (target_line,) = self.ax_joint.plot([], [], "--", linewidth=1.2, label=f"{name}-tgt")
                (actual_line,) = self.ax_joint.plot([], [], "-", linewidth=1.8, label=f"{name}-act")
                self.target_lines.append(target_line)
                self.actual_lines.append(actual_line)
            (self.reward_line,) = self.ax_reward.plot(
                [], [], color="tab:red", linewidth=1.8, label="raw reward"
            )
            self.ax_joint.set_title("Arm Joint Tracking (target vs actual)")
            self.ax_joint.set_ylabel("Joint Angle [rad]")
            self.ax_joint.grid(True, alpha=0.3)
            self.ax_joint.legend(loc="upper right", ncol=3, fontsize=8)
            self.ax_reward.set_title("_reward_arm_manip_commands_tracking_combine")
            self.ax_reward.set_xlabel("Time [s]")
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.grid(True, alpha=0.3)
            self.ax_reward.legend(loc="upper right", fontsize=8)
            self.fig.tight_layout()
            self.enabled = True
        except Exception as exc:
            self.plt = None
            print(f"[plot] matplotlib init failed, disable plotting: {exc}")

    def _set_axis_ylim(self, ax, values, pad_ratio=0.10, min_pad=0.05):
        if not values:
            return
        y_min = min(values)
        y_max = max(values)
        if y_max - y_min < 1e-3:
            y_pad = min_pad
        else:
            y_pad = pad_ratio * (y_max - y_min)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def update(self, step_id, t, target_vals, actual_vals, reward_value):
        if not self.enabled:
            return

        self.t_hist.append(float(t))
        for i in range(len(self.joint_names)):
            self.target_hist[i].append(float(target_vals[i]))
            self.actual_hist[i].append(float(actual_vals[i]))
        self.reward_hist.append(float(reward_value))

        min_t = self.t_hist[-1] - self.history_s
        keep_idx = 0
        while keep_idx < len(self.t_hist) and self.t_hist[keep_idx] < min_t:
            keep_idx += 1
        if keep_idx > 0:
            self.t_hist = self.t_hist[keep_idx:]
            for i in range(len(self.joint_names)):
                self.target_hist[i] = self.target_hist[i][keep_idx:]
                self.actual_hist[i] = self.actual_hist[i][keep_idx:]
            self.reward_hist = self.reward_hist[keep_idx:]

        if step_id % self.update_every != 0:
            return

        for i in range(len(self.joint_names)):
            self.target_lines[i].set_data(self.t_hist, self.target_hist[i])
            self.actual_lines[i].set_data(self.t_hist, self.actual_hist[i])
        self.reward_line.set_data(self.t_hist, self.reward_hist)

        t0 = self.t_hist[0] if self.t_hist else 0.0
        t1 = self.t_hist[-1] if self.t_hist else 1.0
        if t1 <= t0:
            t1 = t0 + 1e-3
        self.ax_joint.set_xlim(t0, t1)
        self.ax_reward.set_xlim(t0, t1)

        flat = []
        for i in range(len(self.joint_names)):
            flat.extend(self.target_hist[i])
            flat.extend(self.actual_hist[i])
        self._set_axis_ylim(self.ax_joint, flat, pad_ratio=0.10, min_pad=0.05)
        self._set_axis_ylim(self.ax_reward, self.reward_hist, pad_ratio=0.10, min_pad=0.02)
        if self.reward_hist:
            reward_latest = self.reward_hist[-1]
            reward_mean = sum(self.reward_hist) / len(self.reward_hist)
            reward_min = min(self.reward_hist)
            self.ax_reward.set_title(
                "_reward_arm_manip_commands_tracking_combine "
                f"(env0, latest={reward_latest:.4f}, mean={reward_mean:.4f}, min={reward_min:.4f})"
            )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.enabled:
            self.plt.ioff()
            self.plt.close(self.fig)
            self.enabled = False


def build_pingpong_indices(num_points):
    if num_points <= 1:
        return [0]
    return list(range(num_points)) + list(range(num_points - 2, 0, -1))


def get_cycle_target(step_idx, waypoints, segment_steps):
    cycle_idx = build_pingpong_indices(waypoints.shape[0])
    seg = max(1, int(segment_steps))
    phase = step_idx // seg
    alpha = float(step_idx % seg) / float(seg)

    i0 = cycle_idx[phase % len(cycle_idx)]
    i1 = cycle_idx[(phase + 1) % len(cycle_idx)]
    return (1.0 - alpha) * waypoints[i0] + alpha * waypoints[i1]


def get_alternating_target(step_idx, waypoints_small, waypoints_large, segment_steps):
    seg = max(1, int(segment_steps))
    phase = step_idx // seg

    idx_small = build_pingpong_indices(waypoints_small.shape[0])
    idx_large = build_pingpong_indices(waypoints_large.shape[0])
    if len(idx_small) != len(idx_large):
        raise ValueError("small/large waypoint pingpong lengths must be identical")

    phases_per_group = len(idx_small)
    group_id = (phase // phases_per_group) % 2  # 0: small, 1: large
    local_phase = phase % phases_per_group
    alpha = float(step_idx % seg) / float(seg)

    if group_id == 0:
        i0 = idx_small[local_phase]
        i1 = idx_small[(local_phase + 1) % len(idx_small)]
        target = (1.0 - alpha) * waypoints_small[i0] + alpha * waypoints_small[i1]
    else:
        i0 = idx_large[local_phase]
        i1 = idx_large[(local_phase + 1) % len(idx_large)]
        target = (1.0 - alpha) * waypoints_large[i0] + alpha * waypoints_large[i1]

    return target, group_id


def get_arm_manip_tracking_reward(env):
    if "arm_manip_commands_tracking_combine" in env.reward_names:
        return float(env.debug_arm_manip_commands_tracking_combine_raw[0].detach().cpu().item())
    with torch.no_grad():
        reward = env.reward_container._reward_arm_manip_commands_tracking_combine()
    return float(reward[0].detach().cpu().item())


def play_go1(args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd, logdir, ckpt_id

    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)

    from go1_gym.utils.global_switch import global_switch

    global_switch.open_switch()

    arm_joint_names = None
    if args.arm_joint_names:
        arm_joint_names = [x.strip() for x in args.arm_joint_names.split(",") if x.strip()]

    env, cfg = load_env(
        logdir,
        wrapper=KeyboardWrapper,
        headless=args.headless,
        device=args.sim_device,
        apply_asset_config_override=True,
        arm_kp=args.arm_kp,
        arm_kd=args.arm_kd,
        arm_joint_names=arm_joint_names,
    )
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)

    env.env.enable_viewer_sync = True
    env.reset()

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd

    arm_start = env.num_actions_loco
    arm_count = env.num_actions_arm
    arm_idx = torch.arange(arm_start, arm_start + arm_count, dtype=torch.long, device=env.device)
    arm_names = [env.dof_names[i] for i in arm_idx.tolist()]

    default_arm = env.default_dof_pos[0, arm_idx].detach().cpu().clone()
    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)

    waypoints_small = torch.tensor(ARM_WAYPOINTS_SMALL_RAD, dtype=torch.float32)
    waypoints_large = torch.tensor(ARM_WAYPOINTS_LARGE_RAD, dtype=torch.float32)
    if waypoints_small.shape[1] != arm_count or waypoints_large.shape[1] != arm_count:
        raise ValueError(
            f"waypoint dimension mismatch: expected {arm_count}, got "
            f"{waypoints_small.shape[1]} and {waypoints_large.shape[1]}"
        )

    plot_joint_num = max(1, min(int(PLOT_NUM_JOINTS), arm_count))
    arm_plotter = None
    if ENABLE_ARM_JOINT_PLOT:
        arm_plotter = ArmTrackingPlotter(
            arm_names[:plot_joint_num],
            history_s=PLOT_HISTORY_S,
            update_every=PLOT_UPDATE_EVERY,
        )
        if "arm_manip_commands_tracking_combine" not in env.reward_names:
            print(
                "[plot] arm_manip_commands_tracking_combine is not active in reward_scales; "
                "the curve will be recomputed directly from the current state."
            )

    zero_plan = torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    try:
        interval_steps = max(1, int(round(PRINT_TRACKING_ERR_INTERVAL_S / float(env.dt))))
        err_sum = torch.zeros(arm_count, dtype=torch.float32)
        err_count = 0
        last_group_id = None

        for step_idx in range(NUM_EVAL_STEPS):
            with torch.no_grad():
                dog_obs = env.get_dog_observations()
                actions_dog = dog_policy(dog_obs).to(env.device)
                env.plan(zero_plan)

            # Absolute desired joint angle from waypoint interpolation [rad].
            target_joint_abs, group_id = get_alternating_target(
                step_idx,
                waypoints_small,
                waypoints_large,
                SEGMENT_STEPS,
            )
            if group_id != last_group_id:
                mode_name = "SMALL" if group_id == 0 else "LARGE"
                print(f"[arm-cycle] switch to {mode_name} amplitude waypoint group")
                last_group_id = group_id
            action_vec = torch.clamp((target_joint_abs - default_arm) / action_scale, -clip_actions, clip_actions)
            # Absolute commanded joint angle actually sent to env [rad].
            cmd_joint_abs = default_arm + action_vec * action_scale

            arm_action = torch.zeros((env.num_envs, arm_count), dtype=torch.float32, device=env.device)
            arm_action[:, :] = action_vec.to(env.device)

            env.step(actions_dog, arm_action)
            manip_reward = get_arm_manip_tracking_reward(env)

            # Tracking error on absolute joint angle [rad].
            actual_joint_abs = env.dof_pos[0, arm_idx].detach().cpu()
            abs_err = torch.abs(cmd_joint_abs - actual_joint_abs)
            err_sum += abs_err
            err_count += 1

            if (step_idx + 1) % interval_steps == 0:
                mean_err = err_sum / max(1, err_count)
                t0 = (step_idx + 1 - err_count) * float(env.dt)
                t1 = (step_idx + 1) * float(env.dt)
                print("=" * 80)
                print(f"[tracking] window {t0:.2f}s -> {t1:.2f}s, mean abs error (rad)")
                for j in range(arm_count):
                    print(f"  {arm_names[j]:12s}: {mean_err[j].item():.4f}")
                print("=" * 80)
                err_sum.zero_()
                err_count = 0

            if arm_plotter is not None:
                arm_plotter.update(
                    step_idx,
                    step_idx * float(env.dt),
                    cmd_joint_abs[:plot_joint_num].numpy(),
                    actual_joint_abs[:plot_joint_num].numpy(),
                    manip_reward,
                )
    finally:
        if arm_plotter is not None:
            arm_plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play by key with arm joint-angle cycle (no arm policy)")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--ckptid", type=int, default=40000)
    parser.add_argument("--arm_kp", type=float, default=None, help="override arm stiffness (kp)")
    parser.add_argument("--arm_kd", type=float, default=None, help="override arm damping (kd)")
    parser.add_argument(
        "--arm_joint_names",
        type=str,
        default="",
        help="comma separated joint names to override, e.g. piper_joint1,piper_joint2",
    )

    args = parser.parse_args()
    play_go1(args)

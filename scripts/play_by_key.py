import time
import isaacgym
import torch
from isaacgym.torch_utils import *
from go1_gym.envs import *
from go1_gym.envs.automatic import KeyboardWrapper
from scripts.load_policy import load_dog_policy, load_arm_policy, load_env
import argparse

logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0

# Realtime plot switch for policy arm-joint targets in play-by-key.
ENABLE_ARM_JOINT_PLOT = True
PLOT_NUM_JOINTS = 6
PLOT_HISTORY_S = 10.0
PLOT_UPDATE_EVERY = 2
PRINT_TRACKING_ERR_INTERVAL_S = 5.0


class ArmJointPlotter:
    def __init__(self, joint_names, history_s=10.0, update_every=2):
        self.joint_names = joint_names
        self.history_s = float(history_s)
        self.update_every = max(1, int(update_every))
        self.enabled = False

        self.t_hist = []
        self.cmd_hist = [[] for _ in joint_names]
        self.act_hist = [[] for _ in joint_names]
        self.reward_hist = []
        self.collision_hist = []

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()
            self.fig, (self.ax_joint, self.ax_reward, self.ax_collision) = self.plt.subplots(
                3,
                1,
                figsize=(10, 8.4),
                sharex=True,
                gridspec_kw={"height_ratios": [3.0, 1.2, 1.0]},
            )
            self.cmd_lines = []
            self.act_lines = []
            for name in joint_names:
                (cmd_line,) = self.ax_joint.plot([], [], "--", linewidth=1.2, label=f"{name}-cmd")
                (act_line,) = self.ax_joint.plot([], [], "-", linewidth=1.8, label=f"{name}-act")
                self.cmd_lines.append(cmd_line)
                self.act_lines.append(act_line)
            (self.reward_line,) = self.ax_reward.plot(
                [], [], color="tab:red", linewidth=1.8, label="raw reward"
            )
            (self.collision_line,) = self.ax_collision.plot(
                [], [], color="tab:purple", linewidth=1.8, label="collision count"
            )
            self.ax_joint.set_title("Arm Joint Command vs Actual (absolute rad)")
            self.ax_joint.set_ylabel("Joint Angle [rad]")
            self.ax_joint.grid(True, alpha=0.3)
            self.ax_joint.legend(loc="upper right", ncol=3, fontsize=8)
            self.ax_reward.set_title("_reward_arm_manip_commands_tracking_combine")
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.grid(True, alpha=0.3)
            self.ax_reward.legend(loc="upper right", fontsize=8)
            self.ax_collision.set_title("_reward_collision raw count")
            self.ax_collision.set_xlabel("Time [s]")
            self.ax_collision.set_ylabel("Count")
            self.ax_collision.grid(True, alpha=0.3)
            self.ax_collision.legend(loc="upper right", fontsize=8)
            self.fig.tight_layout()
            self.enabled = True
        except Exception as exc:
            self.plt = None
            print(f"[plot] matplotlib init failed, disable plotting: {exc}")

    def _set_axis_ylim(self, ax, values, pad_ratio=0.08, min_pad=0.05):
        if not values:
            return
        y_min = min(values)
        y_max = max(values)
        if y_max - y_min < 1e-3:
            y_pad = min_pad
        else:
            y_pad = pad_ratio * (y_max - y_min)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def update(self, step_id, t, cmd_values, act_values, reward_value, collision_count):
        if not self.enabled:
            return

        self.t_hist.append(float(t))
        for i in range(len(self.joint_names)):
            self.cmd_hist[i].append(float(cmd_values[i]))
            self.act_hist[i].append(float(act_values[i]))
        self.reward_hist.append(float(reward_value))
        self.collision_hist.append(float(collision_count))

        min_t = self.t_hist[-1] - self.history_s
        keep_idx = 0
        while keep_idx < len(self.t_hist) and self.t_hist[keep_idx] < min_t:
            keep_idx += 1
        if keep_idx > 0:
            self.t_hist = self.t_hist[keep_idx:]
            for i in range(len(self.joint_names)):
                self.cmd_hist[i] = self.cmd_hist[i][keep_idx:]
                self.act_hist[i] = self.act_hist[i][keep_idx:]
            self.reward_hist = self.reward_hist[keep_idx:]
            self.collision_hist = self.collision_hist[keep_idx:]

        if step_id % self.update_every != 0:
            return

        for i in range(len(self.joint_names)):
            self.cmd_lines[i].set_data(self.t_hist, self.cmd_hist[i])
            self.act_lines[i].set_data(self.t_hist, self.act_hist[i])
        self.reward_line.set_data(self.t_hist, self.reward_hist)
        self.collision_line.set_data(self.t_hist, self.collision_hist)

        t0 = self.t_hist[0] if self.t_hist else 0.0
        t1 = self.t_hist[-1] if self.t_hist else 1.0
        if t1 <= t0:
            t1 = t0 + 1e-3
        self.ax_joint.set_xlim(t0, t1)
        self.ax_reward.set_xlim(t0, t1)
        self.ax_collision.set_xlim(t0, t1)

        flat = []
        for i in range(len(self.joint_names)):
            flat.extend(self.cmd_hist[i])
            flat.extend(self.act_hist[i])
        self._set_axis_ylim(self.ax_joint, flat, pad_ratio=0.08, min_pad=0.05)
        self._set_axis_ylim(self.ax_reward, self.reward_hist, pad_ratio=0.10, min_pad=0.02)
        self._set_axis_ylim(self.ax_collision, self.collision_hist, pad_ratio=0.10, min_pad=0.2)
        if self.reward_hist:
            reward_latest = self.reward_hist[-1]
            reward_mean = sum(self.reward_hist) / len(self.reward_hist)
            reward_min = min(self.reward_hist)
            self.ax_reward.set_title(
                "_reward_arm_manip_commands_tracking_combine "
                f"(env0, latest={reward_latest:.4f}, mean={reward_mean:.4f}, min={reward_min:.4f})"
            )
        if self.collision_hist:
            coll_latest = self.collision_hist[-1]
            coll_mean = sum(self.collision_hist) / len(self.collision_hist)
            coll_max = max(self.collision_hist)
            self.ax_collision.set_title(
                "_reward_collision raw count "
                f"(env0, latest={coll_latest:.1f}, mean={coll_mean:.2f}, max={coll_max:.1f})"
            )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.enabled:
            self.plt.ioff()
            self.plt.close(self.fig)
            self.enabled = False


def get_arm_manip_tracking_reward(env):
    if "arm_manip_commands_tracking_combine" in env.reward_names:
        return float(env.debug_arm_manip_commands_tracking_combine_raw[0].detach().cpu().item())
    with torch.no_grad():
        reward = env.reward_container._reward_arm_manip_commands_tracking_combine()
    return float(reward[0].detach().cpu().item())


def get_collision_count(env):
    with torch.no_grad():
        if env.penalised_contact_indices.numel() == 0:
            return 0.0
        colliding = torch.norm(
            env.contact_forces[:, env.penalised_contact_indices, :], dim=-1
        ) > 0.1
        return float(torch.sum(colliding[0]).detach().cpu().item())


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
        apply_asset_config_override=False,  # already applied in load_policy
        arm_kp=args.arm_kp,
        arm_kd=args.arm_kd,
        arm_joint_names=arm_joint_names,
    )
    dog_policy = load_dog_policy(logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(logdir, ckpt_id, cfg)

    arm_start = env.num_actions_loco
    arm_count = env.num_actions_arm
    arm_idx = torch.arange(arm_start, arm_start + arm_count, dtype=torch.long, device=env.device)
    arm_names = [env.dof_names[i] for i in arm_idx.tolist()]
    plot_joint_num = max(1, min(int(PLOT_NUM_JOINTS), arm_count))
    plot_joint_names = arm_names[:plot_joint_num]

    # Plot-side tensors use CPU to avoid device mismatch with CPU policy outputs.
    default_arm = env.default_dof_pos[0, arm_idx].detach().cpu().clone()
    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)

    arm_plotter = None
    if ENABLE_ARM_JOINT_PLOT:
        arm_plotter = ArmJointPlotter(
            plot_joint_names,
            history_s=PLOT_HISTORY_S,
            update_every=PLOT_UPDATE_EVERY,
        )
        if "arm_manip_commands_tracking_combine" not in env.reward_names:
            print(
                "[plot] arm_manip_commands_tracking_combine is not active in reward_scales; "
                "the curve will be recomputed directly from the current state."
            )
    
    env.env.enable_viewer_sync = True

    num_eval_steps = 30000

    ''' press 'F' to fixed camera'''
    # cam_pos = gymapi.Vec3(4, 3, 2)
    # cam_target = gymapi.Vec3(-4, -3, 0)
    # env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    obs = env.reset()

    env.commands_dog[:, 0] = x_vel_cmd
    env.commands_dog[:, 1] = y_vel_cmd
    env.commands_dog[:, 2] = yaw_vel_cmd
    env.commands_arm[:, 0] = l_cmd
    env.commands_arm[:, 1] = p_cmd
    env.commands_arm[:, 2] = y_cmd
    env.commands_arm[:, 3] = roll_cmd
    env.commands_arm[:, 4] = pitch_cmd
    env.commands_arm[:, 5] = yaw_cmd

    try:
        interval_steps = max(1, int(round(PRINT_TRACKING_ERR_INTERVAL_S / float(env.dt))))
        err_sum = torch.zeros(arm_count, dtype=torch.float32)
        err_count = 0

        obs = env.get_arm_observations()
        for i in (range(num_eval_steps)):
            with torch.no_grad():
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs)
                env.plan(actions_arm[..., -2:])

                dog_obs = env.get_dog_observations()
                actions_dog = dog_policy(dog_obs)

            arm_action = torch.clamp(actions_arm[..., :-2], -clip_actions, clip_actions)
            arm_action_plot = arm_action[0, :arm_count].detach().cpu()
            env.step(actions_dog, arm_action)
            manip_reward = get_arm_manip_tracking_reward(env)
            collision_count = get_collision_count(env)

            arm_joint_cmd_abs = default_arm + arm_action_plot * action_scale
            arm_joint_actual_abs = env.dof_pos[0, arm_idx].detach().cpu()
            abs_err = torch.abs(arm_joint_cmd_abs - arm_joint_actual_abs)
            err_sum += abs_err
            err_count += 1

            if (i + 1) % interval_steps == 0:
                mean_err = err_sum / max(1, err_count)
                t0 = (i + 1 - err_count) * float(env.dt)
                t1 = (i + 1) * float(env.dt)
                print("=" * 80)
                print(f"[tracking] window {t0:.2f}s -> {t1:.2f}s, mean abs error (rad)")
                for j in range(arm_count):
                    print(f"  {arm_names[j]:12s}: {mean_err[j].item():.4f}")
                print("=" * 80)
                err_sum.zero_()
                err_count = 0

            if arm_plotter is not None:
                arm_plotter.update(
                    i,
                    i * float(env.dt),
                    arm_joint_cmd_abs[:plot_joint_num].numpy(),
                    arm_joint_actual_abs[:plot_joint_num].numpy(),
                    manip_reward,
                    collision_count,
                )
    finally:
        if arm_plotter is not None:
            arm_plotter.close()



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    parser = argparse.ArgumentParser(description="Go1")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--ckptid', type=int, default=40000)
    parser.add_argument('--arm_kp', type=float, default=None, help='override arm stiffness (kp)')
    parser.add_argument('--arm_kd', type=float, default=None, help='override arm damping (kd)')
    parser.add_argument(
        '--arm_joint_names',
        type=str,
        default="",
        help='comma separated joint names to override, e.g. piper_joint1,piper_joint2',
    )
   
    args = parser.parse_args()
    play_go1(args)

import time
import isaacgym
import torch
from isaacgym.torch_utils import *
from go1_gym.envs import *
from go1_gym.envs.automatic import KeyboardWrapper
from go1_gym.utils import quaternion_to_rpy
from go1_gym.utils.math_utils import wrap_to_pi
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
PRINT_TRACKING_ERR_INTERVAL_S = 2.0


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
        self.position_error_hist = []
        self.orientation_error_hist = []
        self.orientation_axis_error_hists = [[], [], []]

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()
            self.fig = self.plt.figure(figsize=(12, 11.6))
            gs = self.fig.add_gridspec(
                5,
                3,
                height_ratios=[3.0, 1.1, 1.0, 1.0, 1.0],
            )
            self.ax_joint = self.fig.add_subplot(gs[0, :])
            self.ax_reward = self.fig.add_subplot(gs[1, :], sharex=self.ax_joint)
            self.ax_pos_err = self.fig.add_subplot(gs[2, :], sharex=self.ax_joint)
            self.ax_ori_err = self.fig.add_subplot(gs[3, :], sharex=self.ax_joint)
            self.ax_ori_x = self.fig.add_subplot(gs[4, 0], sharex=self.ax_joint)
            self.ax_ori_y = self.fig.add_subplot(gs[4, 1], sharex=self.ax_joint)
            self.ax_ori_z = self.fig.add_subplot(gs[4, 2], sharex=self.ax_joint)
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
            (self.position_error_line,) = self.ax_pos_err.plot(
                [], [], color="tab:purple", linewidth=1.8, label="position error"
            )
            (self.orientation_error_line,) = self.ax_ori_err.plot(
                [], [], color="tab:green", linewidth=1.8, label="orientation error"
            )
            axis_colors = ["tab:blue", "tab:orange", "tab:brown"]
            axis_titles = ["Roll Error", "Pitch Error", "Yaw Error"]
            axis_labels = ["roll", "pitch", "yaw"]
            self.ax_ori_axes = [self.ax_ori_x, self.ax_ori_y, self.ax_ori_z]
            self.orientation_axis_lines = []
            for ax, color, title, label in zip(self.ax_ori_axes, axis_colors, axis_titles, axis_labels):
                (line,) = ax.plot([], [], color=color, linewidth=1.6, label=label)
                self.orientation_axis_lines.append(line)
                ax.set_title(title)
                ax.set_ylabel("deg")
                ax.set_xlabel("Time [s]")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
            self.ax_joint.set_title("Arm Joint Command vs Actual (absolute rad)")
            self.ax_joint.set_ylabel("Joint Angle [rad]")
            self.ax_joint.set_xlabel("Time [s]")
            self.ax_joint.grid(True, alpha=0.3)
            self.ax_joint.legend(loc="upper right", ncol=3, fontsize=8)
            self.ax_reward.set_title("_reward_arm_manip_commands_tracking_combine")
            self.ax_reward.set_ylabel("Reward")
            self.ax_reward.set_xlabel("Time [s]")
            self.ax_reward.grid(True, alpha=0.3)
            self.ax_reward.legend(loc="upper right", fontsize=8)
            self.ax_pos_err.set_title("End-Effector Position Error")
            self.ax_pos_err.set_ylabel("Error [cm]")
            self.ax_pos_err.set_xlabel("Time [s]")
            self.ax_pos_err.grid(True, alpha=0.3)
            self.ax_pos_err.legend(loc="upper right", fontsize=8)
            self.ax_ori_err.set_title("End-Effector Orientation Error")
            self.ax_ori_err.set_ylabel("Error [deg]")
            self.ax_ori_err.set_xlabel("Time [s]")
            self.ax_ori_err.grid(True, alpha=0.3)
            self.ax_ori_err.legend(loc="upper right", fontsize=8)
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

    def update(
        self,
        step_id,
        t,
        cmd_values,
        act_values,
        reward_value,
        position_error,
        orientation_error,
        orientation_axis_errors,
    ):
        if not self.enabled:
            return

        self.t_hist.append(float(t))
        for i in range(len(self.joint_names)):
            self.cmd_hist[i].append(float(cmd_values[i]))
            self.act_hist[i].append(float(act_values[i]))
        self.reward_hist.append(float(reward_value))
        self.position_error_hist.append(float(position_error))
        self.orientation_error_hist.append(float(orientation_error))
        for i in range(3):
            self.orientation_axis_error_hists[i].append(float(orientation_axis_errors[i]))

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
            self.position_error_hist = self.position_error_hist[keep_idx:]
            self.orientation_error_hist = self.orientation_error_hist[keep_idx:]
            for i in range(3):
                self.orientation_axis_error_hists[i] = self.orientation_axis_error_hists[i][keep_idx:]

        if step_id % self.update_every != 0:
            return

        for i in range(len(self.joint_names)):
            self.cmd_lines[i].set_data(self.t_hist, self.cmd_hist[i])
            self.act_lines[i].set_data(self.t_hist, self.act_hist[i])
        self.reward_line.set_data(self.t_hist, self.reward_hist)
        self.position_error_line.set_data(self.t_hist, self.position_error_hist)
        self.orientation_error_line.set_data(self.t_hist, self.orientation_error_hist)
        for i in range(3):
            self.orientation_axis_lines[i].set_data(self.t_hist, self.orientation_axis_error_hists[i])

        t0 = self.t_hist[0] if self.t_hist else 0.0
        t1 = self.t_hist[-1] if self.t_hist else 1.0
        if t1 <= t0:
            t1 = t0 + 1e-3
        self.ax_joint.set_xlim(t0, t1)
        self.ax_reward.set_xlim(t0, t1)
        self.ax_pos_err.set_xlim(t0, t1)
        self.ax_ori_err.set_xlim(t0, t1)
        self.ax_ori_x.set_xlim(t0, t1)
        self.ax_ori_y.set_xlim(t0, t1)
        self.ax_ori_z.set_xlim(t0, t1)

        flat = []
        for i in range(len(self.joint_names)):
            flat.extend(self.cmd_hist[i])
            flat.extend(self.act_hist[i])
        self._set_axis_ylim(self.ax_joint, flat, pad_ratio=0.08, min_pad=0.05)
        self._set_axis_ylim(self.ax_reward, self.reward_hist, pad_ratio=0.10, min_pad=0.02)
        self._set_axis_ylim(self.ax_pos_err, self.position_error_hist, pad_ratio=0.10, min_pad=0.005)
        self._set_axis_ylim(self.ax_ori_err, self.orientation_error_hist, pad_ratio=0.10, min_pad=0.01)
        for i in range(3):
            self._set_axis_ylim(self.ax_ori_axes[i], self.orientation_axis_error_hists[i], pad_ratio=0.10, min_pad=0.5)
        if self.reward_hist:
            reward_latest = self.reward_hist[-1]
            reward_mean = sum(self.reward_hist) / len(self.reward_hist)
            reward_min = min(self.reward_hist)
            self.ax_reward.set_title(
                "_reward_arm_manip_commands_tracking_combine "
                f"(env0, latest={reward_latest:.4f}, mean={reward_mean:.4f}, min={reward_min:.4f})"
            )
        if self.position_error_hist:
            pos_latest = self.position_error_hist[-1]
            pos_mean = sum(self.position_error_hist) / len(self.position_error_hist)
            pos_max = max(self.position_error_hist)
            self.ax_pos_err.set_title(
                "End-Effector Position Error "
                f"(env0, latest={pos_latest:.4f}, mean={pos_mean:.4f}, max={pos_max:.4f})"
            )
        if self.orientation_error_hist:
            ori_latest = self.orientation_error_hist[-1]
            ori_mean = sum(self.orientation_error_hist) / len(self.orientation_error_hist)
            ori_max = max(self.orientation_error_hist)
            self.ax_ori_err.set_title(
                "End-Effector Orientation Error "
                f"(env0, latest={ori_latest:.4f}, mean={ori_mean:.4f}, max={ori_max:.4f})"
            )
        axis_titles = ["Roll Error", "Pitch Error", "Yaw Error"]
        for i in range(3):
            if self.orientation_axis_error_hists[i]:
                latest = self.orientation_axis_error_hists[i][-1]
                self.ax_ori_axes[i].set_title(f"{axis_titles[i]} (latest error={latest:.2f} deg)")

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


def get_end_effector_errors(env):
    with torch.no_grad():
        env_ids = torch.arange(1, device=env.device)
        actual_lpy = env.get_lpy_in_base_coord(env_ids)
        target_lpy = env.commands_arm_obs[0:1, 0:3]
        actual_xyz = env._arm_lpy_to_local_xyz(actual_lpy)
        target_xyz = env._arm_lpy_to_local_xyz(target_lpy)
        position_error_cm = torch.linalg.norm(actual_xyz - target_xyz, dim=1) * 100.0

        forward = quat_apply(env.base_quat[env_ids], env.forward_vec[env_ids])
        base_yaw = torch.atan2(forward[:, 1], forward[:, 0])
        base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw)
        ee_quat_raw = env.end_effector_state[env_ids, 3:7]
        ee_quat_new = quat_mul(ee_quat_raw, env.ee_rot_offset[env_ids])
        actual_rpy = quaternion_to_rpy(quat_mul(quat_conjugate(base_yaw_quat), ee_quat_new)).to(env.device)
        target_rpy = quaternion_to_rpy(env.obj_quats[0:1]).to(env.device)

        rpy_delta = wrap_to_pi(actual_rpy - target_rpy)
        orientation_error_deg = torch.linalg.norm(rpy_delta, dim=1) * (180.0 / torch.pi)
        orientation_axis_errors_deg = rpy_delta[0] * (180.0 / torch.pi)

        return (
            float(position_error_cm[0].detach().cpu().item()),
            float(orientation_error_deg[0].detach().cpu().item()),
            orientation_axis_errors_deg.detach().cpu().tolist(),
        )


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
            position_error, orientation_error, orientation_axis_errors = get_end_effector_errors(env)

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
                    position_error,
                    orientation_error,
                    orientation_axis_errors,
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

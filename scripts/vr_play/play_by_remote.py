import time

import isaacgym

assert isaacgym
import pickle as pkl

import numpy as np
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.envs.automatic import KeyboardWrapper
from scripts.load_policy import load_dog_policy, load_arm_policy, load_env
from go1_gym.lcm_types.arm_actions_t import arm_actions_t
import math
import threading
import argparse

logdir = "runs/test_roboduet/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

control_type = 'use_key'  # or 'random'
if control_type == 'random':
    moving = False  # random sample velocity
    reorientation = False  # only change orientation with fixd position

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0

shutdown = False
delta_xyzrpy = np.zeros(6)
last_lcm_recv_time = 0.0
lcm_data_seq = 0

import signal
import lcm
lcm_node = lcm.LCM("udpm://239.255.76.67:7136?ttl=255")


class ArmJointPlotter:
    def __init__(self, joint_names, history_s=10.0, update_every=2):
        self.joint_names = joint_names
        self.history_s = float(history_s)
        self.update_every = max(1, int(update_every))
        self.enabled = False

        self.t_hist = []
        self.v_hist = [[] for _ in joint_names]

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(1, 1, figsize=(10, 4))
            self.lines = []
            for name in joint_names:
                (line,) = self.ax.plot([], [], linewidth=1.8, label=name)
                self.lines.append(line)
            self.ax.set_title("Policy Arm Joint Targets (rad)")
            self.ax.set_xlabel("Time [s]")
            self.ax.set_ylabel("Joint Angle [rad]")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(loc="upper right", ncol=2)
            self.enabled = True
        except Exception as exc:
            self.plt = None
            print(f"[plot] matplotlib init failed, disable plotting: {exc}")

    def update(self, step_id, t, values):
        if not self.enabled:
            return

        self.t_hist.append(float(t))
        for i, v in enumerate(values):
            self.v_hist[i].append(float(v))

        min_t = self.t_hist[-1] - self.history_s
        keep_idx = 0
        while keep_idx < len(self.t_hist) and self.t_hist[keep_idx] < min_t:
            keep_idx += 1
        if keep_idx > 0:
            self.t_hist = self.t_hist[keep_idx:]
            for i in range(len(self.v_hist)):
                self.v_hist[i] = self.v_hist[i][keep_idx:]

        if step_id % self.update_every != 0:
            return

        for i, line in enumerate(self.lines):
            line.set_data(self.t_hist, self.v_hist[i])

        t0 = self.t_hist[0] if self.t_hist else 0.0
        t1 = self.t_hist[-1] if self.t_hist else 1.0
        if t1 <= t0:
            t1 = t0 + 1e-3
        self.ax.set_xlim(t0, t1)

        flat = [x for series in self.v_hist for x in series]
        if flat:
            y_min = min(flat)
            y_max = max(flat)
            if y_max - y_min < 1e-3:
                y_pad = 0.05
            else:
                y_pad = 0.08 * (y_max - y_min)
            self.ax.set_ylim(y_min - y_pad, y_max + y_pad)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if self.enabled:
            self.plt.ioff()
            self.plt.close(self.fig)
            self.enabled = False

def arm_data_cb(channel, data):
    global shutdown
    if shutdown:
        return
    global delta_xyzrpy, last_lcm_recv_time, lcm_data_seq
    msg = arm_actions_t.decode(data)
    delta_xyzrpy = np.array(msg.data)[:6]
    last_lcm_recv_time = time.time()
    lcm_data_seq += 1

def signal_handler(sig, frame):
    global shutdown
    shutdown = True

def lcm_thread():
    while not shutdown:
        # Use timeout so the loop can observe shutdown quickly.
        lcm_node.handle_timeout(50)

def play_go1(args):
    
    signal.signal(signal.SIGINT, signal_handler)
    
    vr_control_subscription = lcm_node.subscribe("arm_control_data", arm_data_cb)
    thread1 = threading.Thread(target=lcm_thread, daemon=True)
    thread1.start()
    
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd, delta_xyzrpy, logdir, ckpt_id, last_lcm_recv_time, lcm_data_seq
    
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
    arm_policy = load_arm_policy(logdir, ckpt_id, cfg)

    arm_start = env.num_actions_loco
    arm_count = env.num_actions_arm
    arm_idx = torch.arange(arm_start, arm_start + arm_count, dtype=torch.long, device=env.device)
    arm_names = [env.dof_names[i] for i in arm_idx.tolist()]
    plot_joint_num = max(1, min(int(args.plot_num_joints), arm_count))
    plot_joint_names = arm_names[:plot_joint_num]

    # Plot-side tensors use CPU to avoid device mismatch with CPU policy outputs.
    default_arm = env.default_dof_pos[0, arm_idx].detach().cpu().clone()
    action_scale = float(cfg.control.action_scale)
    clip_actions = float(cfg.normalization.clip_actions)

    arm_plotter = None
    if args.plot_arm_joints:
        arm_plotter = ArmJointPlotter(
            plot_joint_names,
            history_s=args.plot_history_s,
            update_every=args.plot_update_every,
        )
    
    # NOTE: env is a HistoryWrapper; render flags must be set on the wrapped env.
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
    # Keep command observation / visualization in sync with external commands.
    env.env.update_arm_commands()

    default_cmd = np.array([l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float64)
    lcm_stale_timeout_s = 0.3
    
    obs = env.get_arm_observations()
    
    last_arm_actions = None
    last_pitch_roll = None
    filter_rate = 0.8
    pitch_filter_rate = 0.95
    
    try:
        for i in (range(num_eval_steps)):
            if shutdown:
                break

            # Only use remote delta when LCM data is fresh; otherwise keep default command.
            now = time.time()
            has_fresh_lcm = (lcm_data_seq > 0) and ((now - last_lcm_recv_time) <= lcm_stale_timeout_s)
            if has_fresh_lcm:
                delta_x1, delta_y1, delta_z1, delta_roll, delta_pitch, delta_yaw = delta_xyzrpy
                delta_x1 += 0.3
                delta_l = np.sqrt(delta_x1**2 + delta_y1**2 + delta_z1**2)
                delta_y = np.arctan2(delta_y1, delta_x1)
                delta_p = np.arcsin(delta_z1 / delta_l) if delta_l != 0 else 0

                cmd_l = min(max(delta_l + 0.2, 0.3), 0.8)  # 0.3 ~ 0.8
                cmd_p = min(max(delta_p + 0.3, -np.pi/3), np.pi/3)   # -pi/3 ~ pi/3
                cmd_y = min(max(delta_y, -np.pi/2), np.pi/2)  # -pi/2 ~ pi/2

                cmd_alpha = min(max(delta_roll, -np.pi * 0.45), np.pi * 0.45)
                cmd_beta = min(max(delta_pitch, -1.5), 1.5)
                cmd_gamma = min(max(delta_yaw, -1.4), 1.4)
                cmd_alpha, cmd_beta, cmd_gamma = rpy_to_abg(cmd_alpha, cmd_beta, cmd_gamma)

                env.commands_arm[:, 0] = cmd_l
                env.commands_arm[:, 1] = cmd_p
                env.commands_arm[:, 2] = cmd_y
                env.commands_arm[:, 3] = cmd_alpha
                env.commands_arm[:, 4] = cmd_beta
                env.commands_arm[:, 5] = cmd_gamma
            else:
                env.commands_arm[:, 0] = default_cmd[0]
                env.commands_arm[:, 1] = default_cmd[1]
                env.commands_arm[:, 2] = default_cmd[2]
                env.commands_arm[:, 3] = default_cmd[3]
                env.commands_arm[:, 4] = default_cmd[4]
                env.commands_arm[:, 5] = default_cmd[5]

            # Synchronize internal command buffers so the rendered marker follows remote/default target
            # instead of stale sampled targets.
            env.env.update_arm_commands()

            with torch.no_grad():
                obs = env.get_arm_observations()
                actions_arm = arm_policy(obs)

                if last_arm_actions is None:
                    last_arm_actions = actions_arm
                    last_pitch_roll = actions_arm[..., -2:]
                else:
                    last_arm_actions = filter_rate * last_arm_actions + (1 - filter_rate) * actions_arm
                    last_pitch_roll = pitch_filter_rate * last_pitch_roll + (1 - pitch_filter_rate) * actions_arm[..., -2:]

                env.plan(last_pitch_roll)
                dog_obs = env.get_dog_observations()
                actions_dog = dog_policy(dog_obs)

            arm_action = torch.clamp(last_arm_actions[..., :-2], -clip_actions, clip_actions)
            arm_action_plot = arm_action[0, :arm_count].detach().cpu()
            arm_joint_target = default_arm + arm_action_plot * action_scale
            if arm_plotter is not None:
                arm_plotter.update(i, i * float(env.dt), arm_joint_target[:plot_joint_num].numpy())

            ret = env.step(actions_dog, arm_action)
    finally:
        # Best-effort cleanup on normal exit or Ctrl+C
        try:
            lcm_node.unsubscribe(vr_control_subscription)
        except Exception:
            pass
        if arm_plotter is not None:
            arm_plotter.close()


def quat_apply(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def get_rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=-1)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=-1).reshape(shape)

    return quat

def quat_to_angle(quat):
    y_vector = torch.tensor([0., 1., 0.]).double()
    z_vector = torch.tensor([0., 0., 1.]).double()
    x_vector = torch.tensor([1., 0., 0.]).double()
    roll_vec = quat_apply(quat, y_vector) # [0,1,0]
    roll = torch.atan2(roll_vec[2], roll_vec[1]) # roll angle = arctan2(z, y)
    pitch_vec = quat_apply(quat, z_vector) # [0,0,1]
    pitch = torch.atan2(pitch_vec[0], pitch_vec[2]) # pitch angle = arctan2(x, z)
    yaw_vec = quat_apply(quat, x_vector) # [1,0,0]
    yaw = torch.atan2(yaw_vec[1], yaw_vec[0]) # yaw angle = arctan2(y, x)
    
    return torch.stack([roll, pitch, yaw], dim=-1)

def rpy_to_abg(roll, pitch, yaw):
    zero_vec = np.zeros_like(roll)
    q1 = quat_from_euler_xyz(zero_vec, zero_vec, yaw)
    q2 = quat_from_euler_xyz(zero_vec, pitch, zero_vec)
    q3 = quat_from_euler_xyz(roll, zero_vec, zero_vec)
    quats = quat_mul(q1, quat_mul(q2, q3))  # np, (4,)
    abg = quat_to_angle(quats).numpy()
    
    return abg

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
    parser.add_argument('--plot_arm_joints', action='store_true', default=False, help='realtime plot arm target joints')
    parser.add_argument('--plot_num_joints', type=int, default=6, help='number of arm joints to plot')
    parser.add_argument('--plot_history_s', type=float, default=10.0, help='history window length in seconds')
    parser.add_argument('--plot_update_every', type=int, default=2, help='plot redraw interval in steps')
   
    args = parser.parse_args()
    play_go1(args)

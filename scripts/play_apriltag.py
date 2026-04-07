import time
import argparse

import isaacgym

assert isaacgym
import pickle as pkl

import numpy as np
import torch
from isaacgym.torch_utils import *

from go1_gym.envs import *
from go1_gym.utils import quaternion_to_rpy, input_with_timeout
from april_utils import RealTimeCamera
from scripts.load_policy import load_dog_policy, load_arm_policy, load_env
from go1_gym.envs.automatic import KeyboardWrapper


logdir = "/home/a4090/hybrid_improve_dwb/runs/Cooperated_guide_go2/2024-10-13/auto_train/003436.678552_seed9145"
ckpt_id = "040000"

moving = False

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0., 0.0, 0
l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.
roll_cmd, pitch_cmd, yaw_cmd = np.pi/4, np.pi/4, np.pi/4

def play_go1(args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd, logdir, ckpt_id
   
    from go1_gym.utils.global_switch import global_switch
    global_switch.open_switch()

    if args.logdir:
        logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)
    
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
    
    env.enable_viewer_sync = True

    num_eval_steps = 30000

    ''' press 'F' to fixed camera'''
    # cam_pos = gymapi.Vec3(4, 3, 2)
    # cam_target = gymapi.Vec3(-4, -3, 0)
    # env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)

    cam = RealTimeCamera(camera_id=0, tag_type="tag36h11")
    obs = env.reset()
    random = False
    reorientation = False
    obs = env.get_arm_observations()
    # arm_obs = env.get_arm_observations()
    for i in (range(num_eval_steps)):
        
        pose_in_ee = cam.detect()
        pose_in_ee = torch.from_numpy(pose_in_ee, device="cuda:0")
        

        with torch.no_grad():
            t1 = time.time()
            obs = env.get_arm_observations_hand(pose_in_ee)
            actions_arm = arm_policy(obs)
            # print("actions arm: ", actions_arm)
            env.plan(actions_arm[..., -2:])
            dog_obs = env.get_dog_observations_hand(pose_in_ee)
            actions_dog = dog_policy(dog_obs)
            # arm_actions = arm_policy(arm_obs)
            # actions = torch.concat((dog_actions, arm_actions), dim=-1)
        env.commands_dog[:, 0] = x_vel_cmd
        env.commands_dog[:, 1] = 0
        env.commands_dog[:, 2] = yaw_vel_cmd
        # env.commands_dog[:, 10] = -0.4
        # env.commands_dog[:, 11] = 0
        env.commands_arm[:, 0] = l_cmd
        env.commands_arm[:, 1] = p_cmd
        env.commands_arm[:, 2] = y_cmd
        env.commands_arm[:, 3] = roll_cmd
        env.commands_arm[:, 4] = pitch_cmd
        env.commands_arm[:, 5] = yaw_cmd
        env.clock_inputs = 0
        
        ret = env.play(actions_dog, actions_arm[...,:-2], )
        
        if random:
            if i % 100 == 0:
                if not reorientation:
                    l_cmd = torch_rand_float(0.2, 0.8, (1,1), device="cuda:0").squeeze().item()
                    p_cmd = torch_rand_float(-torch.pi/4, torch.pi/4, (1,1), device="cuda:0").squeeze().item()
                    y_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                roll_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                pitch_cmd = torch_rand_float(-torch.pi/3, torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                yaw_cmd = torch_rand_float(-torch.pi/3 , torch.pi/3, (1,1), device="cuda:0").squeeze().item()
                
                quat = quat_from_euler_xyz(torch.tensor(roll_cmd).reshape(-1, 1), torch.tensor(pitch_cmd).reshape(-1, 1), torch.tensor(yaw_cmd).reshape(-1, 1)).reshape(1, 4)
                env.env.obj_quats = quat.to(env.device)
                env.env.visual_rpy = quaternion_to_rpy(quat).to(env.device)
        
        if moving:
            x_vel_cmd = torch_rand_float(0.5, 1, (1,1), device="cuda:0").squeeze().item()
            yaw_vel_cmd = torch_rand_float(-1, 1, (1,1), device="cuda:0").squeeze().item()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go1 AprilTag Play")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--sim_device', type=str, default="cuda:0")
    parser.add_argument('--logdir', type=str, default=logdir)
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

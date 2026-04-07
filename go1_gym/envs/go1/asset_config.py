from typing import Union

from params_proto import Meta

from go1_gym.envs.automatic.legged_robot_config import Cfg

ARM_STIFFNESS_ARM = {
        # "zarx": 50,
        # "zarx_j1": 40,
        # "zarx_j2": 70,
        # "zarx_j3": 70,
        # "zarx_j4": 25,
        # "zarx_j5": 25,
        # "zarx_j6": 25,
        # "zarx_j7": 50,
        # "zarx_j8": 50,
        # "zarx": 50,
        # "piper_joint1": 40,
        # "piper_joint2": 80,
        # "piper_joint3": 80,
        # "piper_joint4": 20,
        # "piper_joint5": 80,
        # "piper_joint6": 20,
        # "piper_joint7": 20,
        # "piper_joint8": 20,
        'piper_joint1': 10,
        'piper_joint2': 10,
        'piper_joint3': 10,
        'piper_joint4': 10,
        'piper_joint5': 10,
        'piper_joint6': 10,
        'piper_joint7': 10,
        'piper_joint8': 10
}  # [N*m/rad]

ARM_DAMPING_ARM = {
        # "zarx": 20,
        # "zarx_j1": 3,
        # "zarx_j2": 15,
        # "zarx_j3": 15,
        # "zarx_j4": 2,
        # "zarx_j5": 2,
        # "zarx_j6": 2,
        # "zarx_j7": 20,
        # "zarx_j8": 20,
        # "zarx": 20,
        # "piper_joint1": 2,
        # "piper_joint2": 4,
        # "piper_joint3": 4,
        # "piper_joint4": 0.5,
        # "piper_joint5": 4,
        # "piper_joint6": 0.5,
        # "piper_joint7": 0.5,
        # "piper_joint8": 0.5,
        'piper_joint1': 1.5,
        'piper_joint2': 1.5,
        'piper_joint3': 1.5,
        'piper_joint4': 1.5,
        'piper_joint5': 1.5,
        'piper_joint6': 1.5,
        'piper_joint7': 1.5,
        'piper_joint8': 1.5
}  # [N*m*s/rad]

ARM_COMMAND_SAFETY = {
    "reject_invalid_targets": True,
    "collision_lower_limits": [-0.38, -0.16, -0.3],
    "collision_upper_limits": [0.3, 0.16, 0.10],
    "underground_limit": -0.38,
    "num_collision_check_samples": 10,
    "resample_max_retries": 10,
}


def apply_arm_pd_from_asset_config(Cnfg: Union[Cfg, Meta]):
    Cnfg.arm.control.stiffness_arm = dict(ARM_STIFFNESS_ARM)
    Cnfg.arm.control.damping_arm = dict(ARM_DAMPING_ARM)


def apply_arm_sampling_safety_from_asset_config(Cnfg: Union[Cfg, Meta]):
    Cnfg.arm.commands.reject_invalid_targets = bool(ARM_COMMAND_SAFETY["reject_invalid_targets"])
    Cnfg.arm.commands.collision_lower_limits = list(ARM_COMMAND_SAFETY["collision_lower_limits"])
    Cnfg.arm.commands.collision_upper_limits = list(ARM_COMMAND_SAFETY["collision_upper_limits"])
    Cnfg.arm.commands.underground_limit = float(ARM_COMMAND_SAFETY["underground_limit"])
    Cnfg.arm.commands.num_collision_check_samples = int(ARM_COMMAND_SAFETY["num_collision_check_samples"])
    Cnfg.arm.commands.resample_max_retries = int(ARM_COMMAND_SAFETY["resample_max_retries"])


def config_asset(Cnfg: Union[Cfg, Meta]):

    Cnfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go2_piper/go2/urdf/go2piper.urdf'
    
    Cnfg.asset.penalize_contacts_on = [
        'base', 'trunk',
        "arm", "wrist", 'zarx',
        "gripper", "thigh", "calf",
        "Head", "piper",
    ]
    
    Cnfg.asset.terminate_after_contacts_on = ['']
    
    Cnfg.asset.hip_joints = {'hip'}
    
    Cnfg.control.stiffness = {'joint': 40., 'widow': 5., "zarx": 5., "zarx_j3": 20. , "piper": 5.}  # [N*m/rad]
    
    
    apply_arm_pd_from_asset_config(Cnfg)
    apply_arm_sampling_safety_from_asset_config(Cnfg)


    Cnfg.dog.control.stiffness_leg = {'joint': 40.}  # [N*m/rad]
    Cnfg.dog.control.damping_leg = {'joint': 1.}  # [N*m*s/rad]
   
    Cnfg.asset.render_sphere = True # NOTE no use in headless 

    Cnfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5,  # [rad]
        
        # 'widow_waist': 0., 
        # 'widow_shoulder': 0., 
        # 'widow_elbow': 0., 
        # 'forearm_roll': 0., 
        # 'widow_wrist_angle': 0., 
        # 'widow_wrist_rotate': 0., 
        # 'widow_forearm_roll': 0., 
        # 'gripper': 0., 
        # 'widow_left_finger': 0., 
        # 'widow_right_finger': 0.,
        
        # "zarx_j1": 0.0,
        # "zarx_j2": 0.8,
        # "zarx_j3": 0.8,
        # "zarx_j4": 0.0,
        # "zarx_j5": 0.0,
        # "zarx_j6": 0.0,
        # "zarx_j7": 0.0,
        # "zarx_j8": 0.0,

        "piper_joint1": 0.0,
        "piper_joint2": 0.6,
        "piper_joint3": -0.5,
        "piper_joint4": 0.0,
        "piper_joint5": 0.0,
        "piper_joint6": 0.0,
        "piper_joint7": 0.0,
        "piper_joint8": 0.0,
        
    }

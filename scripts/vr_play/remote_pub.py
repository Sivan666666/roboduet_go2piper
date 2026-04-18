#!/usr/bin/env python
import lcm
from time import sleep, monotonic
import signal
import sys
import numpy as np
import zmq
import pickle
from datetime import datetime
from pathlib import Path

sys.path.append("../")

from go1_gym.lcm_types.arm_actions_t import arm_actions_t
 
# NOTE This is the ip and port of the pc host connected to vr
GLOBAL_IP = "192.168.1.109"
GLOBAL_PORT = "34567"
lcm_node = lcm.LCM("udpm://239.255.76.67:7136?ttl=255")

LEFT_ARM_FIELD_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]

def calibrate(sock: zmq.Socket, start_pose=None):
    print("Starting calibration. Please keep the controller steady.")
    count = np.zeros(14)
    for _ in range(50):
        action_bin = sock.recv()
        action_ = pickle.loads(action_bin)
        action_ = np.array(action_)
        count += action_[:14]
    count = count / 50.
    print("calibration done")

    x1, y1, z1 = -count[5], -count[3], count[4]
    roll1, pitch1, yaw1 = -count[2], -count[0], count[1]

    x2, y2, z2 = -count[12], -count[10], count[11]
    roll2, pitch2, yaw2 = -count[9], -count[7], count[8]
    

    return np.array([-x1, -y1, -z1, -roll1, -pitch1, -yaw1, 0.,
                        -x2, -y2, -z2, -roll2, -pitch2, -yaw2, 0.]) \
                            + (start_pose if start_pose is not None else 0)


def parser_action(action, offset):

    gripper1 = action[6]
    x1, y1, z1 = -action[5], -action[3], action[4]
    roll1, pitch1, yaw1 = -action[2], -action[0], action[1]

    gripper2 = action[13]
    x2, y2, z2 = -action[12], -action[10], action[11]
    roll2, pitch2, yaw2 = -action[9], -action[7], action[8]

    master_action = np.array([x1, y1, z1, roll1, pitch1, yaw1, gripper1,
                            x2, y2, z2, roll2, pitch2, yaw2, gripper2])
    
    a, b, x, y, thumb_x, thumb_y = action[14], action[15], action[16], action[17], action[18], action[19]
    
    return master_action+offset, a, b, x, y, thumb_x, thumb_y


def get_handles_msg(sock: zmq.Socket, offset):
    action_bin = sock.recv()
    action_ = pickle.loads(action_bin)
    action, a, b, x, y, thumb_x, thumb_y = parser_action(action_, offset)
    return action, a, b, x, y, thumb_x, thumb_y


def format_left_pose_log(pose, delta_t, timestamp_str):
    values = [f"{name}={value:.6f}" for name, value in zip(LEFT_ARM_FIELD_NAMES, pose)]
    dt_text = "N/A" if delta_t is None else f"{delta_t * 1000.0:.3f} ms"
    return f"[{timestamp_str}] left pose | dt={dt_text} | " + ", ".join(values)


def main():
    np.set_printoptions(precision=3)
    arm_pose_msg = arm_actions_t()
    
    start_pos = np.array([0,0,0,0,0,0,0])
    previous_pose = np.zeros(6)
    arm_delta = 0.01

    print("wait for zmq socket connect")
    context = zmq.Context()
    sock = context.socket(zmq.PULL)
    sock.connect(f"tcp://{GLOBAL_IP}:{GLOBAL_PORT}")
    print("socket connected!")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"remote_pub_left_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(
        "# left pose data fields: x, y, z, roll, pitch, yaw\n"
        "# dt: time interval since previous logged left pose\n"
    )
    log_file.flush()
    print(f"left pose log file: {log_path}")
    last_left_log_time = None
    
    # securely close the context
    shutdown = False
    def signal_handler(sig, frame):
        global shutdown
        shutdown = True
        log_file.close()
        sock.close()
        context.term()
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    
    print("press X to start calibration")
    while not shutdown:
        while not shutdown:
            _, a, b, x, y, _, _ = get_handles_msg(sock, np.zeros(14))
            if x == 1:
                offset = calibrate(sock)
                break

        print("press Y to stop teleopration, then press X to reset or press A to exit")
        while not shutdown:
            action, a, b, x, y, thumb_x, thumb_y = get_handles_msg(sock, offset)
            if y == 1:
                while not shutdown:
                    action, a, b, x, y, _,_ = get_handles_msg(sock, offset)
                    if x == 1:
                        arm_pose_msg.data = start_pos[0:6]
                        
                        lcm_node.publish("arm_control_data", arm_pose_msg.encode())

                        break
                    
                    if a == 1:
                        arm_pose_msg.data = start_pos[0:6]
                        
                        lcm_node.publish("arm_control_data", arm_pose_msg.encode())

                        return 0
                break


            if np.sum(np.abs(previous_pose[0:3] - action[0:3])+0.2*np.abs(previous_pose[3:6] - action[3:6])) > arm_delta:
                arm_pose_msg.data = action[0:6]
                previous_pose = action[0:6]
                now_mono = monotonic()
                delta_t = None if last_left_log_time is None else (now_mono - last_left_log_time)
                last_left_log_time = now_mono
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_line = format_left_pose_log(arm_pose_msg.data, delta_t, timestamp_str)
                print(log_line)
                log_file.write(log_line + "\n")
                log_file.flush()
                lcm_node.publish("arm_control_data", arm_pose_msg.encode())


            sleep(0.001)




if __name__ == '__main__':
    main()

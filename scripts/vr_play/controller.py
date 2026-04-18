import sys
import time
import numpy as np

try:
    import openvr
except ImportError:
    print("Please install openvr: pip install openvr")
    sys.exit(1)


# OpenVR frame: +x right, +y up, +z toward the user.
# Robot control frame: +x forward, +y left, +z up.
VR_TO_ROBOT_BASIS = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float32)

class Controller:
    def __init__(self):
        print("Initializing OpenVR System...")
        try:
            # 初始化 SteamVR 
            self.vr_system = openvr.init(openvr.VRApplication_Scene)
        except openvr.OpenVRError as e:
            print(f"Failed to initialize OpenVR: {e}")
            sys.exit(1)
            
        print("OpenVR Tracking Initialized Successfully.")

    def _get_button_state(self, controller_state, button_id):
        return (controller_state.ulButtonPressed & (1 << button_id)) != 0

    @staticmethod
    def _matrix_to_rpy(rot):
        pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
        if np.abs(rot[2, 0]) < 0.999999:
            roll = np.arctan2(rot[2, 1], rot[2, 2])
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-rot[0, 1], rot[1, 1])
        return roll, pitch, yaw

    @staticmethod
    def _vr_pose_to_robot(m):
        rot_vr = np.array([[m[r][c] for c in range(3)] for r in range(3)], dtype=np.float32)
        pos_vr = np.array([m[0][3], m[1][3], m[2][3]], dtype=np.float32)

        pos_robot = VR_TO_ROBOT_BASIS @ pos_vr
        rot_robot = VR_TO_ROBOT_BASIS @ rot_vr @ VR_TO_ROBOT_BASIS.T
        roll, pitch, yaw = Controller._matrix_to_rpy(rot_robot)
        return pos_robot, (roll, pitch, yaw)

    def get_action(self):
        """
        生成器，不断产出大小为 20 的 float 数组并 yield 出去，对齐 remote_pub.py 的 parser_action 解析逻辑：
        [pitch, yaw, roll, y, z, x, trigger] (左手 0-6)
        [pitch, yaw, roll, y, z, x, trigger] (右手 7-13)
        14:A(Right), 15:B(Right), 16:X(Left), 17:Y(Left), 18:joystick_x, 19:joystick_y
        """
        while True:
            # 申请长度20的数组，对应原有的action解包结构
            action = np.zeros(20, dtype=np.float32)
            
            # 获取所有姿态设备 
            poses, _ = openvr.VRCompositor().waitGetPoses(openvr.TrackingUniverseStanding, None)
            
            left_handled, right_handled = False, False

            for i in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class != openvr.TrackedDeviceClass_Controller:
                    continue
                
                # 获取控制器类型（左手还是右手）
                role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)
                
                # 获取按键状态
                result, controller_state = self.vr_system.getControllerState(i)
                if not result:
                    continue

                # 姿态位置信息
                pose = poses[i]
                if not pose.bPoseIsValid:
                    continue

                # 提取矩阵姿态
                m = pose.mDeviceToAbsoluteTracking
                pos_robot, (roll_robot, pitch_robot, yaw_robot) = self._vr_pose_to_robot(m)
                
                # 读取扳机/抓手的值 (Axis 1 is typically trigger)
                trigger = controller_state.rAxis[1].x 

                # 严格对齐 remote_pub.py 的解析顺序：
                # left: [pitch, yaw, roll, y, z, x, trigger] (0-6)
                # right: [pitch, yaw, roll, y, z, x, trigger] (7-13)
                if role == openvr.TrackedControllerRole_LeftHand:
                    # 0:pitch, 1:yaw, 2:roll, 3:y, 4:z, 5:x, 6:trigger
                    # Preserve the historical wire format expected by remote_pub.py.
                    action[0] = -pitch_robot
                    action[1] = yaw_robot
                    action[2] = -roll_robot
                    action[3] = -pos_robot[1]
                    action[4] = pos_robot[2]
                    action[5] = -pos_robot[0]
                    action[6] = trigger
                    
                    # 按键映射：16:X(Left), 17:Y(Left)
                    x_btn = self._get_button_state(controller_state, openvr.k_EButton_A)  # X键
                    y_btn = self._get_button_state(controller_state, openvr.k_EButton_ApplicationMenu)  # Y键
                    action[16] = 1.0 if x_btn else 0.0
                    action[17] = 1.0 if y_btn else 0.0
                    
                    # 摇杆数据：18:joystick_x, 19:joystick_y (Axis 0)
                    action[18] = controller_state.rAxis[0].x
                    action[19] = controller_state.rAxis[0].y
                    
                    left_handled = True

                elif role == openvr.TrackedControllerRole_RightHand:
                    # 7:pitch, 8:yaw, 9:roll, 10:y, 11:z, 12:x, 13:trigger
                    action[7] = -pitch_robot
                    action[8] = yaw_robot
                    action[9] = -roll_robot
                    action[10] = -pos_robot[1]
                    action[11] = pos_robot[2]
                    action[12] = -pos_robot[0]
                    action[13] = trigger
                    
                    # 按键映射：14:A(Right), 15:B(Right)
                    a_btn = self._get_button_state(controller_state, openvr.k_EButton_A)  # A键
                    b_btn = self._get_button_state(controller_state, openvr.k_EButton_ApplicationMenu)  # B键
                    action[14] = 1.0 if a_btn else 0.0
                    action[15] = 1.0 if b_btn else 0.0
                    
                    right_handled = True
            
            # 返回 20 位 float 数组
            yield action
            
            # 控制频率大约 50~100Hz
            time.sleep(0.01)

if __name__ == "__main__":
    c = Controller()
    for d in c.get_action():
        print(np.around(d, 3))

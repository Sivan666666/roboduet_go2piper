import sys
import time
import numpy as np

try:
    import openvr
except ImportError:
    print("Please install openvr: pip install openvr")
    sys.exit(1)

class Controller:
    def __init__(self):
        print("Initializing OpenVR System...")
        try:
            # 初始化 SteamVR 
            self.vr_system = openvr.init(openvr.VRApplication_Background)
        except openvr.OpenVRError as e:
            print(f"Failed to initialize OpenVR: {e}")
            sys.exit(1)
            
        print("OpenVR Tracking Initialized Successfully.")

    def _get_button_state(self, controller_state, button_id):
        return (controller_state.ulButtonPressed & (1 << button_id)) != 0

    def get_action(self):
        """
        生成器，不断产出大小为 20 的 float 数组并 yield 出去，结构：
        [rx, ry, rz, x, y, z, grip] (右/左，依原代码解析器可能左侧映射在前1-7，或是按照左右反着赋值的，这里直接写出14位)
        由于原始解析 parser_action 中:
        1-7 映射为 Left/Right 的姿态和 Trigger
        8-14 映射为另一只手的姿态和 Trigger
        14-19 分别对应 A(Right), B(Right), X(Left), Y(Left), joystick_x, joystick_y
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
                # xyz 位置
                x, y, z = m[0][3], m[1][3], m[2][3]
                
                # 计算 RPY (Roll Pitch Yaw) 从旋转矩阵
                pitch = np.arcsin(max(-1.0, min(1.0, m[2][0])))
                if np.abs(m[2][0]) < 0.999999: # 没遇到万向锁
                    roll = np.arctan2(-m[2][1], m[2][2])
                    yaw = np.arctan2(-m[1][0], m[0][0])
                else: 
                    roll = 0.0
                    yaw = np.arctan2(m[0][1], m[1][1])
                
                # 读取扳机/抓手的值 (Axis 1 is typically trigger)
                trigger = controller_state.rAxis[1].x 

                # 判断左右并赋值给对应数组切片
                # 提示：从 origin parser 来看：
                # count[0] 是 pitch, count[1] 是 yaw, count[2] 是 roll, count[3]是y, count[4]是z, count[5]是x
                
                if role == openvr.TrackedControllerRole_LeftHand:
                    # action[0:7]
                    action[0] = pitch
                    action[1] = yaw
                    action[2] = roll
                    action[3] = y
                    action[4] = z
                    action[5] = x
                    action[6] = trigger
                    
                    # 获取 X, Y 按键
                    # OpenVR 中通常 X 是 7 号(A按键的变体或特定的SteamVR键), Y 是 1 号(或特定的)
                    # 这个对应关系你需要自己在 SteamVR 输入绑定中微调
                    # 此处用标准的 mask 测试
                    x_btn = self._get_button_state(controller_state, openvr.k_EButton_A)
                    y_btn = self._get_button_state(controller_state, openvr.k_EButton_ApplicationMenu)
                    action[16] = 1.0 if x_btn else 0.0
                    action[17] = 1.0 if y_btn else 0.0
                    
                    # 获取 Joystick 数据 (一般 Axis 0)
                    action[18] = controller_state.rAxis[0].x
                    action[19] = controller_state.rAxis[0].y
                    
                    left_handled = True

                elif role == openvr.TrackedControllerRole_RightHand:
                    # action[7:14]
                    action[7] = pitch
                    action[8] = yaw
                    action[9] = roll
                    action[10] = y
                    action[11] = z
                    action[12] = x
                    action[13] = trigger
                    
                    # 获取 A, B 按键
                    a_btn = self._get_button_state(controller_state, openvr.k_EButton_A)
                    b_btn = self._get_button_state(controller_state, openvr.k_EButton_ApplicationMenu)
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
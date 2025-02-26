"""
Control volume by hand gestures (Windows version).

Usage:
    $ python vol_controller.py --control continuous
    or
    $ python vol_controller.py --control step
"""

import argparse
import cv2
import numpy as np
import time

# === 1. 替换 osascript -> pycaw ===
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

from gesture import GestureDetector
from utils.utils import two_landmark_distance, draw_vol_bar, draw_landmarks
from utils.utils import update_trajectory, check_trajectory

CAM_W = 1280
CAM_H = 720
TEXT_COLOR = (102, 51, 0)
ACTI_COLOR = (0, 255, 0)
VOL_RANGE = [0, 100]
BAR_X_RANGE = [50, CAM_W // 5]

def set_volume(volume_level):
    """使用 pycaw 设置系统音量 (0~100)。"""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # 确保音量在 0~100 范围
    volume_level = max(0, min(100, volume_level))
    volume.SetMasterVolumeLevelScalar(volume_level / 100.0, None)

def vol_control(control='continuous', vol_step=10, traj_size=10):
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    ges_detector = GestureDetector(max_num_hands=1)

    # 初始化音量和音量条
    vol = (VOL_RANGE[0] + VOL_RANGE[1]) // 2
    vol_bar = (BAR_X_RANGE[0] + BAR_X_RANGE[1]) // 2

    # === 2. 替换调用 ===
    set_volume(vol)

    ptime = 0
    ctime = 0
    trajectory = []
    target_gestures = ['Pinch', 'C shape']
    wrist, thumb_tip, index_tip = 0, 4, 8
    activated = False
    len_range = None

    while True:
        ret, img = cap.read()
        if not ret:
            print("Could not read from camera. Exiting...")
            break

        img = cv2.flip(img, 1)
        gesture = ges_detector.detect_gesture(img, 'single')
        hands = ges_detector.hand_detector.decoded_hands

        if gesture:
            hand = hands[-1]
            landmarks = hand['landmarks']

            # 如果检测到目标手势则高亮
            if gesture in target_gestures:
                ges_detector.draw_gesture_box(img)

            # Pinch -> 激活手势控制
            if gesture == target_gestures[0]:
                if not activated:
                    base_len = two_landmark_distance(landmarks[wrist], landmarks[thumb_tip])
                    len_range = [0.1 * base_len, 0.6 * base_len]
                    step_threshold = [0.2 * base_len, 0.9 * base_len]
                activated = True

            # C shape -> 取消手势控制
            if activated and gesture == target_gestures[1]:
                activated = False

        if activated and hands:
            hand = hands[-1]
            landmarks = hand['landmarks']
            pt1 = landmarks[thumb_tip][:2]
            pt2 = landmarks[index_tip][:2]
            length = two_landmark_distance(pt1, pt2)

            # === continuous 模式 ===
            if control == 'continuous':
                draw_landmarks(img, pt1, pt2)
                finger_states = ges_detector.check_finger_states(hand)
                # 当拇指与食指尖接近一定程度 (finger_states[4] > 2) 时调节音量
                if finger_states[4] > 2:
                    vol = np.interp(length, len_range, VOL_RANGE)
                    vol_bar = np.interp(length, len_range, BAR_X_RANGE)
                    set_volume(vol)

            # === step 模式 ===
            if control == 'step':
                draw_landmarks(img, pt1, pt2)
                trajectory = update_trajectory(length, trajectory, traj_size)
                up = False
                down = False
                if len(trajectory) == traj_size and length > step_threshold[1]:
                    up = check_trajectory(trajectory, direction=1)
                    if up:
                        vol = min(vol + vol_step, VOL_RANGE[1])
                        set_volume(vol)
                if len(trajectory) == traj_size and length < step_threshold[0]:
                    down = check_trajectory(trajectory, direction=-1)
                    if down:
                        vol = max(vol - vol_step, VOL_RANGE[0])
                        set_volume(vol)
                if up or down:
                    vol_bar = np.interp(vol, VOL_RANGE, BAR_X_RANGE)
                    trajectory = []

        # 计算 FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # 绘制音量条
        pt1 = (30, 20)
        pt2 = (BAR_X_RANGE[1] + 80, 150)
        draw_vol_bar(img, pt1, pt2, vol_bar, vol, fps, BAR_X_RANGE, activated)

        cv2.imshow('Volume controller (Windows)', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=str, default='continuous',
                        help='volume control mode (default: continuous)')
    parser.add_argument('--vol_step', type=int, default=10,
                        help='volume update step for step control (default: 10)')
    parser.add_argument('--traj_size', type=int, default=10,
                        help='trajectory size (default: 10)')
    opt = parser.parse_args()

    vol_control(**vars(opt))

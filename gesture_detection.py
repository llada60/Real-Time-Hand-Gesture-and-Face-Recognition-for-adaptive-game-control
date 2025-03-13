import argparse
import cv2
import time
import numpy as np

import pyautogui
from hand import HandDetector
from utils.templates import Gesture
from utils.utils import two_landmark_distance
from utils.utils import calculate_angle, calculate_thumb_angle, get_finger_state
from utils.utils import map_gesture, draw_bounding_box, draw_fingertips
from gesture_key import GestureControl

THUMB_THRESH = [9, 8]
NON_THUMB_THRESH = [8.6, 7.6, 6.6, 6.1]

BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]

class GestureDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.5):
        
        self.hand_detector = HandDetector(static_image_mode,
                                          max_num_hands,
                                          min_detection_confidence,
                                          min_tracking_confidence)
        self.finger_states = [None] * 5

    
    def check_finger_states(self, hand):
        landmarks = hand['landmarks']
        label = hand['label']
        facing = hand['facing']
        
        joint_angles = np.zeros((5,3)) # 5 fingers and 3 angles

        d1 = two_landmark_distance(landmarks[0], landmarks[5])

        for i in range(5):
            joints = [0, 4*i+1, 4*i+2, 4*i+3, 4*i+4]
            if i == 0:
                joint_angles[i] = np.array(
                    [calculate_thumb_angle(landmarks[joints[j:j+3]], label, facing) for j in range(3)]
                )
                self.finger_states[i] = get_finger_state(joint_angles[i], THUMB_THRESH)
            else:
                joint_angles[i] = np.array(
                    [calculate_angle(landmarks[joints[j:j+3]]) for j in range(3)]
                )
                d2 = two_landmark_distance(landmarks[joints[1]], landmarks[joints[4]])
                self.finger_states[i] = get_finger_state(joint_angles[i], NON_THUMB_THRESH)
                
                if self.finger_states[i] == 0 and d2/d1 < BENT_RATIO_THRESH[i-1]:
                    self.finger_states[i] = 1
        return self.finger_states
    
    def detect_gesture(self, img, num_hands=1, draw=True):
        hands = self.hand_detector.detect_hands(img)
        self.detected_gesture = None

        if hands:
            if num_hands == 1:
                hand = hands[-1]
                self.check_finger_states(hand)
                if draw:
                    self.draw_gesture_landmarks(img)
                print(hand['wrist_angle'])
                # print(hand['landmarks'])
                print(hand['direction'])
                # print(hand['boundary'])
                print("------------------------")
                ges = Gesture(hand['label'])
                self.detected_gesture = map_gesture(ges.gestures,
                                                    self.finger_states,
                                                    hand['landmarks'],
                                                    hand['wrist_angle'],
                                                    hand['direction'],
                                                    hand['boundary'])
            if num_hands == 2 and len(hands) == 2:
                pass

        return self.detected_gesture
    
    def draw_gesture_landmarks(self, img):
        hand = self.hand_detector.decoded_hands[-1]
        self.hand_detector.draw_landmarks(img)
        draw_fingertips(hand['landmarks'], self.finger_states, img)
    
    def draw_gesture_box(self, img):
        hand = self.hand_detector.decoded_hands[-1]
        draw_bounding_box(hand['landmarks'], self.detected_gesture, img)

def main(num_hands=1, target_gesture='all', cam_w=1280, cam_h=720):
    # default module is gaming -- press key based on detected gesture
    # press key 'g' to close/open the gaming module
    # if you want to use dynamic modules, press key 'd'
    cap = cv2.VideoCapture(0)
    cap.set(3, cam_w)
    cap.set(4, cam_h)

    if num_hands>2: num_hands=2
    ges_detector = GestureDetector(max_num_hands=num_hands)
    past_time = 0
    current_time = 0
    gaming_module = True
    dynamic_module = False
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        ges_detector.detect_gesture(img, num_hands)
        if ges_detector.detected_gesture:
            if target_gesture == 'all' or target_gesture == ges_detector.detected_gesture:
                ges_detector.draw_gesture_box(img)
            if gaming_module:
                gesture = GestureControl(ges_detector.detected_gesture) # press key based on detected gesture
                gesture.press_key()
        # compute fps
        current_time = time.time()
        fps = 1 / (current_time - past_time)
        past_time = current_time

        cv2.putText(img, f'FPS: {int(fps)}', (50, 50), 0, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.imshow('Gesture Detection', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        if key == ord('g'):
            gaming_module = not gaming_module
            print(f'Gaming module: {gaming_module}')
        if key == ord('d'):
            dynamic_module = not dynamic_module
            print(f'Dynamic module: {dynamic_module}')
            # lwq TODO: add dynamic control function/class import here
        

    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hands', type=int, default=1,
                        help='Number of hands to detect.')
    parser.add_argument('--target_gesture', type=str, default='all',
                        help='Target gesture to detect. (default: all)')
    parser.add_argument('--cam_w', type=int, default=1280,
                        help='Camera width. (default: 1280)')
    parser.add_argument('--cam_h', type=int, default=720,
                        help='Camera height. (default: 720)')
    args = parser.parse_args()
    main(**vars(args))
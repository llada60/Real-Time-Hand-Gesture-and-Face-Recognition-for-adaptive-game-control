import argparse
import cv2
import time
import numpy as np

import random
from hand import HandDetector
from utils.templates import Gesture
from utils.utils import two_landmark_distance
from utils.utils import calculate_angle, calculate_thumb_angle, get_finger_state
from utils.utils import map_gesture, draw_bounding_box, draw_fingertips
from gesture_key import GestureControl
import math



THUMB_THRESH = [9, 8]
NON_THUMB_THRESH = [8.6, 7.6, 6.6, 6.1]

BENT_RATIO_THRESH = [0.76, 0.88, 0.85, 0.65]
#-----------------------------
#彩带
#-----------------------------
class ConfettiParticle:
    def __init__(self, screen_width, screen_height):

        self.pos = np.array([random.randint(0, screen_width), random.randint(-50, 0)], dtype=float)

        self.vel = np.array([random.uniform(-1, 1), random.uniform(1, 3)], dtype=float)  
        self.color = tuple(np.random.randint(100, 256, size=3).tolist())

    
        self.length = random.randint(30, 80)  
        self.width = random.randint(6, 14) 

        self.wave_amplitude = random.uniform(5, 15)  
        self.wave_frequency = random.uniform(0.1, 0.3)  


        self.angle = random.uniform(0, 360) 
        self.life = random.randint(80, 150)
        self.initial_life = self.life

        self.curve_dir = random.choice([-1, 1]) 
    def update(self):
        self.life -= 1
        self.pos += self.vel
        self.pos[0] += math.sin(self.pos[1] * self.wave_frequency) * self.wave_amplitude 

    def is_alive(self):
        return self.life > 0

    def draw(self, img):

        alpha = (self.life / self.initial_life) ** 2 
    
        start_point = (int(self.pos[0]), int(self.pos[1]))
        end_point = (
            int(self.pos[0] + math.cos(math.radians(self.angle)) * self.length),
            int(self.pos[1] + math.sin(math.radians(self.angle)) * self.length),
        )

        control_point1 = (
            start_point[0] + self.curve_dir * self.width,
            start_point[1] + self.width // 2
        )
        control_point2 = (
            end_point[0] + self.curve_dir * self.width,
            end_point[1] - self.width // 2
        )

        pts = np.array([start_point, control_point1, end_point, control_point2], np.int32)
        pts = pts.reshape((-1, 1, 2))

        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], self.color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class ConfettiEffect:
    def __init__(self, screen_width, screen_height):
        self.particles = []
        self.screen_width = screen_width
        self.screen_height = screen_height

    def update(self, spawn_new=False):
        self.particles = [p for p in self.particles if p.is_alive()]
        if spawn_new:
            for _ in range(5):  
                self.particles.append(ConfettiParticle(self.screen_width, self.screen_height))
        for p in self.particles:
            p.update()

    def draw(self, img):
        for p in self.particles:
            p.draw(img)

# -----------------------------
#  烟花特效类
# -----------------------------
class Particle:
    def __init__(self, pos, velocity=None, color=None):
        self.pos = np.array(pos, dtype=float)
        if velocity is None:
            vx = random.uniform(-3, 3)
            vy = random.uniform(-4, -1)  
            self.vel = np.array([vx, vy], dtype=float)
        else:
            self.vel = np.array(velocity, dtype=float)
        
        self.life = random.randint(30, 50)
        self.initial_life = self.life
        
        if color is None:

            self.color = tuple(np.random.randint(50, 256, size=3).tolist())
        else:
            self.color = color
        self.radius = random.randint(4, 10)
    
    def update(self):
        self.life -= 1   
        self.pos += self.vel
        self.vel[1] += 0.15

    def is_alive(self):
        return self.life > 0
    
    def draw(self, img):
        alpha = (self.life / self.initial_life) ** 2  
        draw_radius = max(1, int(self.radius * alpha))
        draw_color = (
            int(self.color[0] * alpha),
            int(self.color[1] * alpha),
            int(self.color[2] * alpha),
        )
        if draw_radius > 0:
            center = (int(self.pos[0]), int(self.pos[1]))
            cv2.circle(img, center, draw_radius, draw_color, -1)



class FireworkEffect:
    def __init__(self):
        self.particles = []

    def update(self, position=None, spawn_new=False):
        self.particles = [p for p in self.particles if p.is_alive()]


        if spawn_new and position is not None:
            for _ in range(10):  
                offset_x = random.uniform(-10, 10)  
                offset_y = random.uniform(-10, 10)
                spawn_pos = (position[0] + offset_x, position[1] + offset_y)

                self.particles.append(Particle(spawn_pos))

        for p in self.particles:
            p.update()

    def draw(self, img):
        for p in self.particles:
            p.draw(img)


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
                # print(hand['wrist_angle'])
                # # print(hand['landmarks'])
                # print(hand['direction'])
                # # print(hand['boundary'])
                # print("------------------------")
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
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Actual camera resolution:", actual_width, actual_height)

    if num_hands>2: num_hands=2
    ges_detector = GestureDetector(max_num_hands=num_hands)
    past_time = 0
    current_time = 0
    gaming_module = 0 # 0 no gaming; 1 space jump; 2 arrow keys; 3 both
    dynamic_module = False
    
    firework_effect = FireworkEffect()
    confetti_effect = ConfettiEffect(actual_width, actual_height)
    while True:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        ges_detector.detect_gesture(img, num_hands)
        # print(ges_detector.detected_gesture)
        finger_tip_pixel = None
        h, w, _ = img.shape
  
        if dynamic_module:
            if ges_detector.hand_detector.decoded_hands and ges_detector.detected_gesture != 'Thumbs-up':           
                finger_tip_pixel = ges_detector.hand_detector.decoded_hands[-1]['landmarks'][8][:2]
                firework_effect.update(position=finger_tip_pixel, spawn_new=True)
            else:
                firework_effect.update(spawn_new=False)


            firework_effect.draw(img)

        if ges_detector.detected_gesture:
            spawn_confetti = ges_detector.detected_gesture == 'Thumbs-up'
            confetti_effect.update(spawn_new=spawn_confetti)
            confetti_effect.draw(img)
            if target_gesture == 'all' or target_gesture == ges_detector.detected_gesture:
                ges_detector.draw_gesture_box(img)
            if gaming_module:
                gesture = GestureControl(ges_detector.detected_gesture) # press key based on detected gesture
                gesture.press_key(gaming_module)
        
        if dynamic_module:
            print("Finger tip pos:", finger_tip_pixel)
            for particle in firework_effect.particles:
                 color = particle.color
                 cv2.circle(img, tuple(particle.pos.astype(int)), 8, color, -1)

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
            gaming_module = (gaming_module + 1) % 4
            if gaming_module == 0:
                print("Gaming module is off")
            elif gaming_module == 1:
                print("Dinosaur Gaming module: C Shape for Space jump, Palm Horizontal for Duck")
            elif gaming_module == 2:
                print("Pacman Gaming module: Arrow keys")
            else:
                print("Gaming module: Space jump and Arrow keys")
        if key == ord('d'):
            dynamic_module = not dynamic_module
            print(f'Dynamic module: {dynamic_module}')
            # lwq TODO: add dynamic control function/class import here
        
    cap.release()
    cv2.destroyAllWindows()
    
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
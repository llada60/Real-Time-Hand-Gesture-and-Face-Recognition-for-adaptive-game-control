
# Real-time Hand Gesture and Face Recognition for adative Game Control

This project aims to develop a real-time interaction system for gaming applications and interactive visual effects using state-of-the-art computer vision techniques. It consists of two main components: hand gesture recognition and facial expression recognition.

![pipeline_of_project](https://github.com/llada60/Real-Time-Hand-Gesture-and-Face-Recognition-for-adaptive-game-control/blob/main/img/main.png)

Hand gesture recognition based on the hand module of Google's [mediapipe](https://github.com/google/mediapipe) API. The hand module gives the coordinates of 21 hand landmarks, which can be found in the image below.

![hand_landmark](https://github.com/llada60/Real-Time-Hand-Gesture-and-Face-Recognition-for-adaptive-game-control/blob/main/img/hand_landmark.png)

This project focuses on four functionalities:
1. Hand gesture recognition.
2. Facial expression recognition.
3. Game control by keyboard using hand gestures and facial expressions.
4. Visual effects using hand gestures.

## Requirements
Python 3.8 or later with dependencies listed in requirements.txt. To install run:

```bash
$ git clone https://github.com/llada60/Real-Time-Hand-Gesture-and-Face-Recognition-for-adaptive-game-control.git
$ cd Real-Time-Hand-Gesture-and-Face-Recognition-for-adaptive-game-control
$ pip install -r requirements.txt
```

## Usage

```bash
# Hand Recognition
$ python gesture_detection.py
```

For different game controls, you can press 'g' to switch different game modules. The following modules are available:
1. [Dinosaur Gaming](https://chromedino.com/) Module: c shape for space jump and palm horizontal for duck.
2. [Pacman Gaming](https://www.google.com/logos/2010/pacman10-i.html) Module: different turning hand to control arrow keys.
3. Both arrow keys and space key.
4. Turn off gaming module.

For visual module, you can press 'd' for open or close.

## Demo

### Game Control with hand gestures

### Visual Effects with hand gestures

### Game Control with facial expressions

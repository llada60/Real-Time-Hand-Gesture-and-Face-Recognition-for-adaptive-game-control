import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyautogui
import time

model_path = r"C:\Users\sdkd1\Desktop\LECTURE\CV\Project\Real-Time-Hand-Gesture-Recognition-for-Adaptive-Game-Control\fer_mlp_model.h5"  # 替换为你的模型路径
model = tf.keras.models.load_model(model_path)
class_labels = ["angry","disgust","fear","happy","neutral","sad","surprise"]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_468_coords(landmarks):
    if len(landmarks) > 468:
        landmarks = landmarks[:468]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
    if coords.shape[0] != 468 * 3:
        raise ValueError(f"size of landmarks is wrong,expect 1404,current {coords.shape[0]}")
    return coords

COOLDOWN_TIME = 5.0  
last_trigger_time = {"angry": 0, "happy": 0}

cap = cv2.VideoCapture(0)
print("pree 'q' exit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    label_text = "No face"
    if results.multi_face_landmarks:
        try:
            face_landmarks = results.multi_face_landmarks[0]
            coords = extract_468_coords(face_landmarks.landmark)
            
            
            coords_reshaped = coords.reshape(1, -1).astype(np.float32)
            preds = model.predict(coords_reshaped, verbose=0)
            class_idx = np.argmax(preds)
            confidence = preds[0][class_idx]
            
            current_time = time.time()
            if class_labels[class_idx] == "angry":
                if current_time - last_trigger_time["angry"] >= COOLDOWN_TIME:
                    pyautogui.press('s')
                    last_trigger_time["angry"] = current_time
                label_text = f"angry (lower complexity) ({confidence:.2f})"
            elif class_labels[class_idx] == "happy":
                if current_time - last_trigger_time["happy"] >= COOLDOWN_TIME:
                    pyautogui.press('w')
                    last_trigger_time["happy"] = current_time
                label_text = f"happy (higher complexity) ({confidence:.2f})"
            else:
                label_text = f"{class_labels[class_idx]} ({confidence:.2f})"
            
            xs = [lm.x for lm in face_landmarks.landmark[:468]]
            ys = [lm.y for lm in face_landmarks.landmark[:468]]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_left = int(min_x * w)
            bbox_right = int(max_x * w)
            bbox_top = int(min_y * h)
            bbox_bottom = int(max_y * h)
            cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), (0,255,0), 2)
            
            for i, lm in enumerate(face_landmarks.landmark[:468]):
                if i % 30 == 0:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (0,255,0), -1)
                    
        except Exception as e:
            label_text = f"Error: {str(e)}"
            print("[DEBUG] 异常:", str(e))
    
    cv2.putText(frame, f"Emotion: {label_text}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("current emotion", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 1: 加载手势识别模型,可识别21种手势
base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
print("Gesture recognizer model loaded successfully!")

# STEP 2: 打开摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # STEP 3: 将 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # STEP 4: 识别手势
    recognition_result = recognizer.recognize(mp_image)

    # STEP 5: 显示结果
    if recognition_result.gestures and len(recognition_result.gestures) > 0:
        top_gesture = recognition_result.gestures[0][0]
        gesture_category = top_gesture.category_name
        score = top_gesture.score
        cv2.putText(frame, f"Gesture: {gesture_category} ({score:.2f})", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No gesture detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
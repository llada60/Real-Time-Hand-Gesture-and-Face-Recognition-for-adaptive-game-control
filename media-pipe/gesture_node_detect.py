import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils  # 用于画关键点

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 处理帧（BGR → RGB）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行手部检测
    results = hands.process(frame_rgb)

    # 如果检测到手
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 画出手部关键点
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 获取21个关键点坐标
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 显示结果
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 头部倾斜角度阈值 
ANGLE_THRESHOLD = 180


def calculate_angle(a,b,c):
    radians = np.arctan2(c.y-b.y, c.x-b.x) - np.arctan2(a.y-b.y, a.x-b.x)
    angle = radians*180.0/np.pi
    if angle < 0:
        angle = 360+angle
    return angle


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # 重新格式化图像
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 检测姿态
        results = pose.process(image)
        
        # 重新格式化回BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 绘制姿态 landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:
            # 检查眼睛角度
            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            left_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
            left_eye_inner = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
            
            left_angle = calculate_angle(left_ear, left_eye_outer, left_eye_inner)
            cv2.putText(image, f'left: {left_angle}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            right_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
            right_eye_inner = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
            
            right_angle = calculate_angle(right_eye_inner, right_eye_outer, right_ear)
            cv2.putText(image, f'right: {right_angle}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if left_angle > ANGLE_THRESHOLD or right_angle > ANGLE_THRESHOLD:
                cv2.putText(image, 'raise up your head!', (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()

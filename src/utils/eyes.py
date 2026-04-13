import cv2
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]


def crop_eye_from_landmarks(frame, landmarks, eye_indices, pad=8):
    h, w = frame.shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_indices])
    x, y, bw, bh = cv2.boundingRect(pts)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    crop = frame[y0:y1, x0:x1]
    return crop
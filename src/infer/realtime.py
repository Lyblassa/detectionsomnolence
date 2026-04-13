# src/infer/realtime.py
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

from src.config import ALARM_SOUND, BEST_MODEL_PATH
from src.utils.alarms import AlarmPlayer
from src.utils.eyes import crop_eye_from_landmarks, RIGHT_EYE, LEFT_EYE
from src.ui.overlay import draw_hud, red_alert_overlay

mp_face_mesh = mp.solutions.face_mesh

# ---- Réglages de sensibilité (faciles à comprendre) ----
THRESH = 0.45          # probabilité "yeux fermés" (>= THRESH = fermé)
CLOSED_SECONDS = 4.0   # durée (en secondes) de fermeture CONTINUE avant alarme


def preprocess_eye(img_bgr, target_w, target_h, need_channel=True):
    """BGR -> Gray -> resize -> [0,1] -> shape (1, H, W, 1) si need_channel=True."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (target_w, target_h)) / 255.0
    if need_channel:
        gray = gray.astype(np.float32)[None, ..., None]   # (1, H, W, 1)
    else:
        gray = gray.astype(np.float32)[None, ...]         # (1, H, W)
    return gray


def main():
    # Charge le modèle et récupère (H, W, C) d'entrée
    model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    _, H, W, C = in_shape
    need_channel = (C is not None and C == 1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera introuvable")

    alarm = AlarmPlayer(ALARM_SOUND)

    # Compteur de temps de fermeture continue
    closed_time = 0.0
    last_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            prob_closed = 0.0

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark

                # Recadre les deux yeux
                eye_l = crop_eye_from_landmarks(frame, lms, LEFT_EYE)
                eye_r = crop_eye_from_landmarks(frame, lms, RIGHT_EYE)

                # Préprocess + prédiction pour chaque œil
                batch = []
                for eye in (eye_l, eye_r):
                    arr = preprocess_eye(eye, target_w=W, target_h=H, need_channel=need_channel)
                    if arr is not None:
                        batch.append(arr)

                if batch:
                    X = np.concatenate(batch, axis=0)              # (k, H, W, 1)
                    preds = model.predict(X, verbose=0).flatten()  # prob "closed" pour chaque œil
                    # On prend le MAX des deux yeux (plus strict: si un œil est clairement fermé)
                    prob_closed = float(np.max(preds))

            # ---- Logique "fermeture CONTINUE pendant N secondes" ----
            if prob_closed >= THRESH:
                closed_time += dt
            else:
                closed_time = 0.0

            # HUD: on affiche la prob, le seuil et "secondes fermées"/"objectif"
            # (j'utilise les 2 derniers champs du HUD pour indiquer temps cumulé vs. objectif)
            draw_hud(frame, prob_closed, THRESH, int(round(closed_time)), int(CLOSED_SECONDS))

            # Déclenchement si fermeture continue >= CLOSED_SECONDS
            if closed_time >= CLOSED_SECONDS:
                red_alert_overlay(frame)
                alarm.start()
            else:
                alarm.stop()

            cv2.imshow("Drowsiness (CNN)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

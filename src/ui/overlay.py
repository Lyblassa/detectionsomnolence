import cv2
import numpy as np

def draw_hud(frame, prob_closed, threshold, consec, max_consec):
    text = f"Closed prob: {prob_closed:.2f} | THR: {threshold:.2f} | FRAMES>=: {consec}/{max_consec}"
    cv2.rectangle(frame, (10, 10), (10 + 500, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def red_alert_overlay(frame):
    overlay = frame.copy()
    overlay[:] = (0, 0, 255)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "DROWSINESS DETECTED!", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

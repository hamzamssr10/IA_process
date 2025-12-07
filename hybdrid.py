import cv2
import numpy as np
import time
from collections import deque
import requests


def calculate_tenengrad(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = cv2.magnitude(sx, sy)
    return float(mag.mean())

def calculate_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def smooth_score(score, history, alpha=0.3):
    if len(history) == 0:
        return score
    return alpha * score + (1 - alpha) * history[-1]

def calculate_hybrid_focus(frame, lap_history, ten_history):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap_score = calculate_laplacian(gray)
    ten_score = calculate_tenengrad(gray)

    lap_smooth = smooth_score(lap_score, lap_history)
    ten_smooth = smooth_score(ten_score, ten_history)

    hybrid_score = 0.5 * lap_smooth + 0.5 * ten_smooth
    return hybrid_score

rtsp_link = "rtsp://admin:2899100*-+@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"





focus_threshold = 1200

def monitor_focus():
    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    lap_history = deque(maxlen=10)
    ten_history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)
        attempts = 0
        increase_focus()
        ret, frame = cap.read()
        if not ret:
            break

        hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)
        attempts += 1
        
        print("Switching to decreasing focus")
        hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)


        status = f"Focus Score: {hybrid_score:.1f}"

        time.sleep(0.05)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor_focus()

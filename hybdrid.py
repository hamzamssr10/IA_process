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


FOCUS_SERVER = "http://192.168.25.25:5000"


def _focus_move(direction: str):
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/move",
            json={"direction": direction},
            timeout=0.5,
        )
    except requests.RequestException as e:
        print(f"Error sending focus move ({direction}): {e}")


def focus_stop():
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/stop",
            timeout=0.5,
        )
    except requests.RequestException as e:
        print(f"Error sending focus stop: {e}")


def increase_focus():
    _focus_move("in")
    focus_stop()


def decrease_focus():
    _focus_move("out")
    focus_stop()

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
        while hybrid_score < focus_threshold and attempts < 30:
            increase_focus()
            ret, frame = cap.read()
            if not ret:
                break

            hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)
            attempts += 1
        
        if hybrid_score < focus_threshold:
            print("Switching to decreasing focus")
            attempts = 0
            while hybrid_score < focus_threshold and attempts < 30:
                decrease_focus()
                ret, frame = cap.read()
                if not ret:
                    break

                hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)
                attempts += 1


        status = f"Focus Score: {hybrid_score:.1f}"

        # cv2.putText(frame, f"Laplacian: {lap_smooth:.1f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # cv2.putText(frame, f"Tenengrad: {ten_smooth:.1f}", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(frame, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        time.sleep(0.05)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor_focus()

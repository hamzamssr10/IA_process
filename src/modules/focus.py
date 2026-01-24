"""Auto-focus module for camera focus adjustment."""
import cv2
import time
import threading
import requests
from collections import deque
from .config import FOCUS_SERVER


def calculate_tenengrad(gray):
    """Calculate Tenengrad focus measure."""
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag = cv2.magnitude(sx, sy)
    return float(mag.mean())


def calculate_laplacian(gray):
    """Calculate Laplacian variance for focus measure."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def smooth_score(score, history, alpha=0.3):
    """Apply exponential smoothing to focus score."""
    if len(history) == 0:
        return score
    return alpha * score + (1 - alpha) * history[-1]


def calculate_hybrid_focus(frame, lap_history, ten_history):
    """Calculate hybrid focus score combining Laplacian and Tenengrad."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap_score = calculate_laplacian(gray)
    ten_score = calculate_tenengrad(gray)

    lap_smooth = smooth_score(lap_score, lap_history)
    ten_smooth = smooth_score(ten_score, ten_history)

    lap_history.append(lap_smooth)
    ten_history.append(ten_smooth)

    hybrid_score = 0.5 * lap_smooth + 0.5 * ten_smooth
    return hybrid_score


def _focus_move(direction, cam):
    """Send focus move command to camera."""
    try:
        requests.post(
            f"{FOCUS_SERVER}/camera/{cam}/ptz/focus/{direction}",
            json={"channel": 1},
            timeout=0.5,
        )
    except requests.RequestException as e:
        print(f"Error sending focus move: {e}")


def _focus_stop(cam):
    """Send focus stop command to camera."""
    try:
        requests.post(
            f"{FOCUS_SERVER}/camera/{cam}/ptz/focus/stop",
            json={"channel": 1},
            timeout=0.5,
        )
    except requests.RequestException as e:
        print(f"Error sending focus stop: {e}")


def increase_focus(cam, duration: float = 1):
    """Move focus far."""
    _focus_move("far", cam)
    time.sleep(duration)
    _focus_stop(cam)


def decrease_focus(cam, duration: float = 1):
    """Move focus near."""
    _focus_move("near", cam)
    time.sleep(duration)
    _focus_stop(cam)


def frame_reader(video_source, shared, lock, stop_event):
    """Read frames for focus monitoring."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[FOCUS] Video source not accessible: {video_source}")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if not video_source.startswith('rtsp'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.02)
            continue

        with lock:
            shared["frame"] = frame

    cap.release()


def focus_controller(cam, shared, lock, stop_event, step=1):
    """Control focus adjustment based on focus metrics."""
    lap_history = deque(maxlen=10)
    ten_history = deque(maxlen=10)

    prev_score = None
    direction = "increase"
    reversed_once = False
    stable_c = 0
    EPS = 5.0

    while not stop_event.is_set():
        with lock:
            frame = shared.get("frame")

        if frame is None:
            time.sleep(0.01)
            continue

        score = calculate_hybrid_focus(frame, lap_history, ten_history)

        if prev_score is None:
            prev_score = score
            print(f"[FOCUS] {cam} initial score: {score:.2f}")
            increase_focus(cam, step)
            continue

        if score > prev_score + EPS:
            if direction == "increase":
                increase_focus(cam, step)
            else:
                decrease_focus(cam, step)

        elif score < prev_score - EPS:
            if direction == "increase" and not reversed_once:
                direction = "decrease"
                reversed_once = True
                print(f"[FOCUS] {cam} switching → decrease")
                decrease_focus(cam, step)
            else:
                print(f"[FOCUS] {cam} focus locked")
                stop_event.set()
                break

        else:
            if stable_c >= 3:
                print(f"[FOCUS] {cam} focus locked (stable)")
                stop_event.set()
                break
            stable_c += 1
            if direction == "increase":
                increase_focus(cam, step)
            else:
                decrease_focus(cam, step)

        prev_score = score
        time.sleep(0.2)


def monitor_focus(cam, video_source, stop_event, step=1):
    """Monitor and adjust camera focus."""
    print(f"[FOCUS] {cam} start monitoring")

    shared = {"frame": None}
    lock = threading.Lock()

    t_reader = threading.Thread(
        target=frame_reader,
        args=(video_source, shared, lock, stop_event),
        daemon=True
    )

    t_focus = threading.Thread(
        target=focus_controller,
        args=(cam, shared, lock, stop_event, step),
        daemon=True
    )

    t_reader.start()
    t_focus.start()

    t_focus.join()
    stop_event.set()
    t_reader.join()

    print(f"[FOCUS] {cam} monitoring stopped")

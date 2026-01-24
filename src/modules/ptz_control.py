"""PTZ (Pan-Tilt-Zoom) control module."""
import time
import requests
from .config import FOCUS_SERVER


def send_ptz_request(direction, cam, speed=4):
    """Send PTZ movement request."""
    url = f"{FOCUS_SERVER}/camera/{cam}/ptz/move/{direction}"
    payload = {"speed": speed}
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code == 200:
            print(f"[PTZ] {cam} moved {direction} successfully")
        else:
            print(f"[PTZ] Request failed ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"[PTZ] error: {e}")


def send_ptz_request_stop(cam, speed=4):
    """Send PTZ stop request."""
    url = f"{FOCUS_SERVER}/camera/{cam}/ptz/move/stop"
    payload = {"speed": speed}
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code == 200:
            print(f"[PTZ] {cam} stopped successfully")
        else:
            print(f"[PTZ] Request failed ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"[PTZ] stop error: {e}")


def cam_left(cam, speed=4):
    """Move camera left."""
    send_ptz_request("left", cam, speed)
    time.sleep(0.1 + speed * 0.02)
    send_ptz_request_stop(cam)


def cam_right(cam, speed=4):
    """Move camera right."""
    send_ptz_request("right", cam, speed)
    time.sleep(0.1 + speed * 0.02)
    send_ptz_request_stop(cam)


def cam_up(cam, speed=4):
    """Move camera up."""
    send_ptz_request("up", cam, speed)
    time.sleep(0.1 + speed * 0.02)
    send_ptz_request_stop(cam)


def cam_down(cam, speed=4):
    """Move camera down."""
    send_ptz_request("down", cam, speed)
    time.sleep(0.1 + speed * 0.02)
    send_ptz_request_stop(cam)


def cam_stop(cam):
    """Stop camera movement."""
    send_ptz_request_stop(cam)


def compute_ptz_speed(error, dead_zone, min_speed=2, max_speed=8):
    """Compute PTZ speed based on error distance."""
    abs_error = abs(error)

    if abs_error <= dead_zone:
        return 0

    norm = min(abs_error / (dead_zone * 4), 1.0)
    speed = min_speed + norm * (max_speed - min_speed)
    return int(round(speed))


def ptz_follow(cam, track_id, cx, cy, frame, frame_cx, frame_cy, target_id):
    """Follow a tracked object with PTZ camera."""
    import cv2
    
    DEAD_ZONE = 80

    x1 = frame_cx - DEAD_ZONE
    y1 = frame_cy - DEAD_ZONE
    x2 = frame_cx + DEAD_ZONE
    y2 = frame_cy + DEAD_ZONE

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    if track_id != target_id:
        return frame

    error_x = cx - frame_cx
    error_y = cy - frame_cy

    speed_x = compute_ptz_speed(error_x, DEAD_ZONE)
    speed_y = compute_ptz_speed(error_y, DEAD_ZONE)

    if abs(error_x) > DEAD_ZONE:
        cam_right(cam, speed_x) if error_x > 0 else cam_left(cam, speed_x)

    if abs(error_y) > DEAD_ZONE:
        cam_down(cam, speed_y) if error_y > 0 else cam_up(cam, speed_y)

    if abs(error_x) <= DEAD_ZONE and abs(error_y) <= DEAD_ZONE:
        cam_stop(cam)

    return frame

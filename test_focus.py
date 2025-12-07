import requests
import time

# Server URL
FOCUS_SERVER = "http://localhost:3000"
RTSP_LINK = "rtsp://admin:2899100*-+@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"  # <-- replace with your RTSP URL

def _focus_move(direction: str):
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/move",
            json={"direction": direction,"time":1,"speed":5},
            timeout=0.5,
        )
        print(f"Focus move sent: {direction}")
    except requests.RequestException as e:
        print(f"Error sending focus move ({direction}): {e}")

# def focus_stop():
#     try:
#         requests.post(
#             f"{FOCUS_SERVER}/focus/cam2/stop",
#             timeout=1,
#         )
#         print("Focus stop sent")
#     except requests.RequestException as e:
#         print(f"Error sending focus stop: {e}")

def increase_focus():
    _focus_move("focus_in")
    # focus_stop()

def decrease_focus():
    _focus_move("focus_out")
    # focus_stop()

# -------------------------------
# Test with video display
# -------------------------------

while True:
        
        print("Increasing focus...")
        increase_focus()
        time.sleep(2)
        decrease_focus()
        time.sleep(2)

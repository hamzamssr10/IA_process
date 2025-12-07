import requests
import cv2
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

def focus_stop():
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/stop",
            timeout=1,
        )
        print("Focus stop sent")
    except requests.RequestException as e:
        print(f"Error sending focus stop: {e}")

def increase_focus():
    _focus_move("focus_in")
    # focus_stop()

def decrease_focus():
    _focus_move("focus_out")
    # focus_stop()

# -------------------------------
# Test with video display
# -------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(RTSP_LINK)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        exit()

    print("Press 'i' to increase focus, 'd' to decrease, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break
        print("Increasing focus...")
        increase_focus()
        time.sleep(0.5)
        # Show the frame
        cv2.imshow("Focus Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            focus_stop()
            break

    cap.release()
    cv2.destroyAllWindows()

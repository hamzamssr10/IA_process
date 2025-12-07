import requests

# Server URL
FOCUS_SERVER = "http://localhost:3000"

def _focus_move(direction: str):
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/move",
            json={"direction": direction},
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
# Example test
# -------------------------------
if __name__ == "__main__":
    import time

    print("Testing focus control...")

    print("Increasing focus for 2 seconds...")
    increase_focus()
    time.sleep(2)
    focus_stop()

    time.sleep(1)

    print("Decreasing focus for 2 seconds...")
    decrease_focus()
    time.sleep(2)
    focus_stop()

    print("Test finished.")

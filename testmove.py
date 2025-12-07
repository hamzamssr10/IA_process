import requests
import time

# Replace this with your actual server URL
FOCUS_SERVER = "http://localhost:3000"

def send_ptz_request(direction, speed=8, channel=0, duration=200):
    url = f"{FOCUS_SERVER}/ptz/cam2/move"
    payload = {
        "direction": direction,
        "speed": speed,
        "channel": channel,
        "duration": duration
    }
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code != 200:
            print(f"PTZ request failed: {r.status_code}")
        else:
            print(f"PTZ request sent successfully: {direction}")
    except Exception as e:
        print(f"PTZ request error: {e}")

def send_ptz_request2(direction, speed=8, channel=0, duration=200):
    url = f"{FOCUS_SERVER}/ptz/cam2/stop"
    payload = {
        "direction": direction,
        "speed": speed,
        "channel": channel,
        "duration": duration
    }
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code != 200:
            print(f"PTZ request failed: {r.status_code}")
        else:
            print(f"PTZ request sent successfully: {direction}")
    except Exception as e:
        print(f"PTZ request error: {e}")

if __name__ == "__main__":
    directions = ["up", "down", "left", "right"]
    
    for dir in directions:
        print(f"Sending PTZ request: {dir}")
        send_ptz_request(direction=dir)
        time.sleep(1)  # wait 1s between requests
        send_ptz_request2(direction=dir)

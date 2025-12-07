import cv2
import time

RTSP_URL = "rtsp://admin:2899100*-+@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

print("✅ RTSP stream opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # draw timestamp so you see live updates
    cv2.putText(frame, str(time.time()), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("RTSP Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

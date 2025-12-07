from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import cv2
from ultralytics import YOLO
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
import threading
import json
from types import SimpleNamespace
import socket
import cv2
import numpy as np
from datetime import datetime
from threading import Lock
from collections import deque
import requests
import os 
from typing import List, Optional
import time
import uvicorn



sock = None
dest_ip = "255.255.255.255"
dest_port = 5012


app = FastAPI()


class ObjectData(BaseModel):
    CLS: str
    ID_TRACK: str
    X: int
    Y: int
    Z: Optional[int] = 0


class DetectionData(BaseModel):
    len: int
    data: List[ObjectData]

def calculate_crc16(data):
    """
    Calcule le CRC16 (CCITT) pour validation

    Args:
        data: bytearray des données

    Returns:
        int: CRC16 (2 bytes)
    """
    crc = 0xFFFF
    polynomial = 0x1021

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ polynomial
            else:
                crc = crc << 1
            crc &= 0xFFFF

    return crc



def start_udp(ip="255.255.255.255", port=5005):
    """Démarre le socket UDP en mode broadcast"""
    global sock, dest_ip, dest_port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        dest_ip = ip
        dest_port = port
        print(f"✓ UDP Socket démarré en mode BROADCAST pour {dest_ip}:{dest_port}")
        return True
    except Exception as e:
        print(f"✗ Erreur démarrage UDP: {e}")
        return False


def convert_and_send(json_data):
    """
    Convertit les données JSON et envoie via UDP

    Args:
        json_data: Dictionnaire avec 'len' et 'data'

    Returns:
        bool: True si envoi réussi, False sinon
    """
    global sock, dest_ip, dest_port

    if not sock:
        print("✗ Socket UDP non démarré.")
        return False

    try:
        # Construction du paquet
        data = bytearray()

        # Header
        data.append(0xFB)

        # Nombre d'objets
        nb_objects = json_data["len"]
        data.append(nb_objects)

        # Boucle sur chaque objet
        for obj in json_data["data"]:
            # Conversion des valeurs hexadécimales string en int
            cls = int(obj["CLS"], 16) if isinstance(obj["CLS"], str) else obj["CLS"]
            id_track = (
                int(obj["ID_TRACK"], 16)
                if isinstance(obj["ID_TRACK"], str)
                else obj["ID_TRACK"]
            )

            # Conversion X, Y et Z en hexadécimal (4 bytes chacun)
            x_hex = obj["X"] & 0xFFFFFFFF
            y_hex = obj["Y"] & 0xFFFFFFFF
            z_hex = obj.get("Z", 0) & 0xFFFFFFFF

            # Ajout au paquet
            data.append(cls & 0xFF)
            data.append(id_track & 0xFF)

            # X (4 bytes - big endian)
            data.append((x_hex >> 24) & 0xFF)
            data.append((x_hex >> 16) & 0xFF)
            data.append((x_hex >> 8) & 0xFF)
            data.append(x_hex & 0xFF)

            # Y (4 bytes - big endian)
            data.append((y_hex >> 24) & 0xFF)
            data.append((y_hex >> 16) & 0xFF)
            data.append((y_hex >> 8) & 0xFF)
            data.append(y_hex & 0xFF)

            # Z (4 bytes - big endian)
            data.append((z_hex >> 24) & 0xFF)
            data.append((z_hex >> 16) & 0xFF)
            data.append((z_hex >> 8) & 0xFF)
            data.append(z_hex & 0xFF)

            print(
                f"  Objet: CLS=0x{cls:02X}, ID=0x{id_track:02X}, X=0x{x_hex:08X}({obj['X']}), Y=0x{y_hex:08X}({obj['Y']}), Z=0x{z_hex:08X}({obj.get('Z', 0)})"
            )

        # Calcul du CRC16 sur toutes les données
        crc = calculate_crc16(data)

        # Ajout du CRC à la fin (2 bytes - big endian)
        data.append((crc >> 8) & 0xFF)  # CRC high byte
        data.append(crc & 0xFF)  # CRC low byte

        print(f"\n   CRC16: 0x{crc:04X}")

        # Envoi
        sock.sendto(data, (dest_ip, dest_port))

        print(f"\n Envoyé {len(data)} bytes: {' '.join(f'{b:02X}' for b in data)}")
        return True

    except Exception as e:
        print(f"✗ Erreur envoi: {e}")
        return False


def stop_udp():
    """Ferme le socket UDP"""
    global sock
    if sock:
        sock.close()
        sock = None
        print("✓ UDP Socket fermé")


@app.on_event("startup")
async def startup_event():
    """Démarre le socket UDP au démarrage de l'API"""
    start_udp(ip="255.255.255.255", port=5012)


@app.on_event("shutdown")
async def shutdown_event():
    """Ferme le socket UDP à l'arrêt de l'API"""
    stop_udp()





def update_buffer(frame, BUFFER):
    BUFFER.append(frame.copy())

lock = Lock()
BUFFER = deque(maxlen=150)
BUFFER_t = deque(maxlen=150)
BUFFER_obj = {}  

SAVE_BASE = "recordings"
os.makedirs(SAVE_BASE, exist_ok=True)

def save_clip(path, BUFFER, fps=30):
    if len(BUFFER) < 5:
        return
    h, w = BUFFER[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in list(BUFFER):
        writer.write(f)
    writer.release()


def save_data(full_frame1, full_frame2, obj, BUFFER_full,BUFFER_full_2, BUFFER_obj, event_type="event"):
    with lock:

        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S_%f")
        BUFFER_obj_1 = BUFFER_obj[0]
        BUFFER_obj_2 = BUFFER_obj[1]

        class_name = obj["class_name"]
        track_id = obj["id"]
        x1, y1, x2, y2 = map(int, obj["bbox"])
        crop_1 = full_frame1[y1:y2, x1:x2]
        crop_2 = full_frame2[y1:y2, x1:x2]

        base_dir = os.path.join(
            SAVE_BASE,
            now.strftime("%Y-%m-%d"),
            str(class_name),
            f"track_{track_id}",
            event_type
        )
        os.makedirs(base_dir, exist_ok=True)

        frame_path1 = os.path.join(base_dir, f"frame1_{ts}.jpg")
        cv2.imwrite(frame_path1, full_frame1)

        frame_path2 = os.path.join(base_dir, f"frame2_{ts}.jpg")
        cv2.imwrite(frame_path2, full_frame2)

        crop_path_1 = os.path.join(base_dir, f"crop_{ts}_1.jpg")
        if crop_1.size > 0:
            cv2.imwrite(crop_path_1, crop_1)
        else:
            crop_path_1 = None

        crop_path_2 = os.path.join(base_dir, f"crop_{ts}_2.jpg")
        if crop_2.size > 0:
            cv2.imwrite(crop_path_2, crop_2)
        else:
            crop_path_1 = None

        full_clip_path = os.path.join(base_dir, f"full_clip_{ts}.mp4")
        save_clip(full_clip_path, BUFFER_full)

        full_clip_path_2 = os.path.join(base_dir, f"full_clip_2_{ts}.mp4")
        save_clip(full_clip_path_2, BUFFER_full_2)

        obj_clip_path_1 = os.path.join(base_dir, f"object_clip_{ts}_1.mp4")
        save_clip(obj_clip_path_1, BUFFER_obj_1)

        obj_clip_path_2 = os.path.join(base_dir, f"object_clip_{ts}_2.mp4")
        save_clip(obj_clip_path_2, BUFFER_obj_2)

        meta_path = os.path.join(base_dir, f"meta_{ts}.json")
        metadata = {
            "timestamp": ts,
            "event": event_type,
            "class": class_name,
            "track_id": track_id,
            "bbox": obj["bbox"],
            "frame_path1": frame_path1,
            "frame_path2": frame_path2,
            "crop_path_1": crop_path_1,
            "crop_path_2": crop_path_2,
            "full_clip": full_clip_path,
            "object_clip_1": obj_clip_path_1,
            "object_clip_2": obj_clip_path_2
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[SAVE] Event saved → {base_dir}")







RTSP_RGP = "rtsp://admin:2899100*-+@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
RTSP_THER = "rtsp://admin:2899100*-+@192.168.1.109:554/cam/realmonitor?channel=1&subtype=0"

args = SimpleNamespace(
    track_high_thresh=0.6,
    track_low_thresh=0.1,
    new_track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True
)
tracker = BYTETracker(args, frame_rate=30)


process_thread_f = None
stop_flag_f = False


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

    lap_history.append(lap_smooth)
    ten_history.append(ten_smooth)

    hybrid_score = 0.5 * lap_smooth + 0.5 * ten_smooth
    return hybrid_score

FOCUS_SERVER = "http://localhost:3000"

#def _focus_move(direction: str, speed: int = 2, time: float = 0.4):
def _focus_move(payload):
    try:
        requests.post(
            f"{FOCUS_SERVER}/focus/cam2/move",
            json= payload,
            timeout=0.5,
        )
        print(f"Focus move sent: {payload}")
    except requests.RequestException as e:
        print(f"Error sending focus move : {e}")

def _focus_stop(payload):
    try:
        result = requests.post(
            f"{FOCUS_SERVER}/focus/cam2/stop",
            json=payload,
            timeout=0.5,
        )
        print(f"Focus stop sent: {payload}   {result} ")
    except requests.RequestException as e:
        print(f"Error sending focus move : {e}")

def increase_focus(speed: int = 2, duration: float = 0.5):
    """Move focus in - longer duration for meaningful movement"""
    pyload = {"direction": "focus_in", "speed": speed}
    _focus_move(pyload)
    time.sleep(duration)  # Allow actual movement
    _focus_stop(pyload)

def decrease_focus(speed: int = 2, duration: float = 0.2):
    """Move focus out - longer duration for meaningful movement"""
    pyload = {"direction": "focus_out", "speed": speed}
    _focus_move(pyload)
    time.sleep(duration)  # Allow actual movement
    _focus_stop(pyload)



def monitor_focus(rtsp_link=RTSP_RGP):
    focus_threshold = 1800
    print("Start monitoring focus...")
    cap = cv2.VideoCapture(rtsp_link)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return
    
    lap_history = deque(maxlen=10)
    ten_history = deque(maxlen=10)
    
    
    # Get initial score
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        return
    
    hybrid_score = calculate_hybrid_focus(frame, lap_history, ten_history)
    print(f"Initial hybrid score: {hybrid_score:.2f}")
    
    if hybrid_score + 100 < focus_threshold:
        prev_score = hybrid_score
        direction = "increase"  # Start by increasing

        consecutive_worse = 0  # Track how many times score got worse
        
        while hybrid_score < focus_threshold: # and attempts < max_attempts:
            # Calculate dynamic parameters based on distance
            score_diff = focus_threshold - hybrid_score
            speed, duration = 2, 0.2
            
            print(f"Score: {hybrid_score:.2f} | Target: {focus_threshold} | Diff: {score_diff:.2f}")
            print(f"Direction: {direction} | Speed: {speed} | Duration: {duration}s")
            
            # Adjust focus with dynamic parameters
            if direction == "increase":
                increase_focus(speed, duration)
            else:
                decrease_focus(speed, duration)
            
            # Wait for camera to settle
            time.sleep(0.5)
            
            # Update history with fresh frames
            for _ in range(3):
                ret, frame = cap.read()
                if not ret:
                    break
                calculate_hybrid_focus(frame, lap_history, ten_history)
                time.sleep(0.05)
            
            if not ret:
                print("Frame capture failed")
                break
            
            # Get new score
            ret, frame = cap.read()
            if not ret:
                break
                
            new_score = calculate_hybrid_focus(frame, lap_history, ten_history)
            score_change = new_score - prev_score
            
            print(f"New score: {new_score:.2f} | Change: {score_change:+.2f}")
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Score: {new_score:.2f}/{focus_threshold}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Dir: {direction} | Speed: {speed}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(display_frame, f"Change: {score_change:+.2f}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Focus Monitor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Decision logic: only reverse if consistently getting worse
            if new_score > prev_score:
                # Score improved!
                print("✓ Score improved! Continuing same direction...")
                consecutive_worse = 0
            else:
                # Score got worse
                consecutive_worse += 1
                print(f"✗ Score worsened ({consecutive_worse} times)")
                
                # Only reverse after 2-3 consecutive bad moves
                if consecutive_worse >= 2:
                    direction = "decrease" if direction == "increase" else "increase"
                    consecutive_worse = 0
                    print(f"→ Reversing direction to {direction}")
            
            prev_score = new_score
            hybrid_score = new_score
        
        print(f"Focus adjustment complete!")
        print(f"Final score: {hybrid_score:.2f} | Target: {focus_threshold}")
        print(f"Attempts: {attempts}")
        print(f"{'='*50}")
    else:
        print(f"Focus already above threshold: {hybrid_score:.2f}")
    
    
    cap.release()
    cv2.destroyAllWindows()

class FakeResults:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.xywh = np.zeros_like(xyxy)
        self.xywh[:,0] = xyxy[:,0]
        self.xywh[:,1] = xyxy[:,1]
        self.xywh[:,2] = xyxy[:,2] - xyxy[:,0]
        self.xywh[:,3] = xyxy[:,3] - xyxy[:,1]
        self.conf = conf
        self.cls = cls

    def __getitem__(self, idx):
        return FakeResults(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.conf)

def make_fake_yolo_results(dets: np.ndarray):
    if dets.shape[0] == 0:
        return FakeResults(np.empty((0,4)), np.empty((0,)), np.empty((0,)))
    xyxy = dets[:, :4]
    conf = dets[:, 4]
    cls = np.zeros(len(dets))
    return FakeResults(xyxy, conf, cls)


model_path = "yolov8s.pt"
model_yolo = YOLO(model_path)

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=25, detectShadows=True
)




def track_objects_yolo(model, frame):
    results = model.track(frame,conf = 0.3, persist=True, verbose=False)
    final_output = []

    if len(results) == 0:
        return final_output

    det = results[0]
    boxes = det.boxes
    if boxes is None:
        return final_output

    for box in boxes:
        bbox = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        conf =  float(box.conf[0])

        final_output.append({
            "id": str(track_id),
            "class_name": cls,
            "bbox": bbox
        })

    return final_output





bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=32,
    detectShadows=False
)

def detect_motion(frame, min_area=300):
    fg = bg_sub.apply(frame)

    fg = cv2.GaussianBlur(fg, (3, 3), 0)

    _, mask = cv2.threshold(fg, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + w, y + h
        motion_boxes.append([x1, y1, x2, y2])

    return motion_boxes, mask


process_thread = None
stop_flag = False

def iou_xyxy(a, b):
    """Compute IoU where boxes are [x1,y1,x2,y2]"""
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union if union > 0 else 0.0

def get_motion_only_objects(motion_boxes, yolo_detections, iou_thresh=0.2):
    motion_only = []
    for m_box in motion_boxes:
        found = False
        for det in yolo_detections:
            y_box = det["bbox"]  # already xyxy
            if iou_xyxy(m_box, y_box) > iou_thresh:
                found = True
                break
        if not found:
            motion_only.append(m_box)
    return motion_only




def merge_motion_into_yolo(yolo_results, motion_only):

    for box in motion_only:
        x1, y1, x2, y2, track_id = box[:5]

        fake_det = {
            "id": str(track_id) + "m",
            "class_name": 81,
            "bbox": [abs(int(x1)), abs(int(y1)),abs( int(x2)), abs(int(y2))],
        }

        yolo_results.append(fake_det)
    return yolo_results


def motion_only_with_conf(motion_only, conf=0.85):
    return [[x1, y1, x2, y2, conf] for x1, y1, x2, y2 in motion_only]




def string_to_hex(s):
    return s.encode("utf-8").hex()

def process_data(data):
    list_of = []
    for obj in data:
        x1 , y1, x2, y2  = obj["bbox"]
        X , Y  = int((x1+x2)/2), int((y1 + y2)/2)
        ID_TRACK =  string_to_hex(obj["id"])
        CLS = hex(obj["class_name"])
        list_of.append({"CLS": CLS, "ID_TRACK":ID_TRACK, "X":X, "Y":Y, "Z": 0})
    return list_of

def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0,0,3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]

backup  = []

memo  = {}

def IA_process(rtsp_RGB = RTSP_RGP, rtsp_thermique = RTSP_THER):
    global stop_flag
    global backup
    global memo 

    
    cap_rgb = cv2.VideoCapture(rtsp_RGB)
    cap_ther = cv2.VideoCapture(rtsp_thermique)

    while True and not stop_flag:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_ther, frame_ther = cap_ther.read()

        if not ret_rgb and ret_ther:
            break
        
        update_buffer(frame_rgb, BUFFER)
        update_buffer(frame_ther,BUFFER_t)

        save_all = True
        if save_all:
            path_all = os.path.join(SAVE_BASE, "all_events")
            os.makedirs(path_all, exist_ok=True)
            save_clip(os.path.join(path_all, f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"),
                      BUFFER)
            save_clip(os.path.join(path_all, f"clip_t_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"),
                      BUFFER_t)

        yolo_results = track_objects_yolo(model_yolo, frame_rgb)

        motion_boxes, _ = detect_motion(frame_rgb)
        motion_only = get_motion_only_objects(motion_boxes, yolo_results)
        motion_only = motion_only_with_conf(motion_only)
        motion_only  = np.array(motion_only) if motion_only else np.empty((0,5))
        fake_results = make_fake_yolo_results(motion_only)
        tracks = tracker.update(fake_results, frame_rgb)

        final_results = merge_motion_into_yolo(yolo_results, tracks)
        print("final_results :",final_results)
        if len(final_results) > 0 :
            # send events
            data_event = process_data(final_results)
            #backup = data_event + backup

            STATIC_DATA = {"len" : len(data_event) , "data" : data_event}
            print(STATIC_DATA)
            convert_and_send(STATIC_DATA)
            
            # save events 
            for object in final_results:
                track_id_t = object["id"]
                cls_name_t = object["class_name"]
                key = f"{track_id_t}-{cls_name_t}"

                if key not in BUFFER_obj:
                    BUFFER_obj[key] = [deque(maxlen=150),deque(maxlen=150)]

                box =  list(map(int , object["bbox"]))
                x1, y1, x2, y2 = box
                
                x_c, y_c = int((x1 + x2) / 2), int((y1 + y2) / 2)

                track_key = string_to_hex(track_id_t)
                
                if track_key not in memo:
                    memo[track_key] = []
                
                memo[track_key].append((x_c, y_c))

                if len(memo[track_key]) > 50:
                    memo[track_key].pop(0)

                cropped_rgb = crop(frame_rgb,box)
                cropped_ther = crop(frame_ther,box)

                update_buffer(cropped_rgb, BUFFER_obj[key][0])
                update_buffer(cropped_ther,BUFFER_obj[key][1])

                save_data(frame_rgb, frame_ther, object, BUFFER,BUFFER_t, BUFFER_obj[key])
                # ---- DRAW BOXES ----
                x1, y1, x2, y2 = box
                # drawing the tracking lines ... 
                for track_key, points in memo.items():
                    if len(points) > 1:
                        for i in range(1, len(points)):
                            cv2.line(
                                frame,
                                points[i - 1],
                                points[i],
                                (0, 255, 0),  
                                2              
                            )
                color = (0, 255, 0)  # green box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_rgb, f"{cls_name_t}-{track_id_t}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame_ther, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_ther, f"{cls_name_t}-{track_id_t}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the frames
        cv2.imshow("RGB Stream", frame_rgb)
        #cv2.imshow("Thermal Stream", frame_ther)

            

    cap_rgb.release()
    cap_ther.release()
    cv2.destroyAllWindows()



@app.post("/detection/start")
async def start_process():
    global process_thread, stop_flag
    if process_thread and process_thread.is_alive():
        return {"status": "already running"}

    stop_flag = False
    process_thread = threading.Thread(target=IA_process, daemon=True)
    process_thread.start()
    return {"status": "processing started"}

@app.post("/detection/stop")
async def stop_process():
    global stop_flag
    stop_flag = True
    return {"status": "stopping processing"}


@app.post("/focus/start")
async def start_process_focus():
    global process_thread_f, stop_flag_f
    if process_thread_f and process_thread_f.is_alive():
        return {"status": "already running"}

    stop_flag_f = False
    process_thread_f = threading.Thread(target=monitor_focus, daemon=True)
    process_thread_f.start()
    return {"status": "Focus started"}

class TrackRequest(BaseModel):
    id: str

@app.post("/track/object")
async def track_object(request: TrackRequest):
    global memo, backup
    
    target_hex = hex(request.id)   # convert int → hex string
    
    # Check if this ID exists in memo
    if target_hex in memo:
        positions = memo[target_hex]
        last_pos = positions[-1] if positions else None
        
        return {
            "status": "found",
            "id": request.id,
            "hex_id": target_hex,
            "last_position": last_pos,
            "path_length": len(positions)
        }
    
    # Not found → return last global status
    last_status = backup[-1] if backup else None
    
    return {
        "status": "not found",
        "id": request.id,
        "last_status": last_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9898
    )


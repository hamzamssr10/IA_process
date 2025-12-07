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
from typing import List, Optional, Set, Tuple
import time
import uvicorn
import asyncio


# Variables globales
sock = None
dest_port = 5012
sending_task = None
listener_task = None

# Liste des clients connectés (IP, Port)
connected_clients: Set[Tuple[str, int]] = set()
app = FastAPI()

# Modèle Pydantic pour validation
class ObjectData(BaseModel):
    CLS: str
    ID_TRACK: str
    X: int
    Y: int
    Z: Optional[int] = 0


class DetectionData(BaseModel):
    len: int
    data: List[ObjectData]

def calculate_checksum(data):
    """
    Calcule le checksum simple (somme des données seulement, sans header ni checksum)
    Checksum = (Byte2 + Byte3 + Byte4 + ... + ByteN) & 0xFF

    Args:
        data: bytearray des données (incluant le header, sans le checksum final)

    Returns:
        int: Checksum (1 byte)
    """
    # Somme de tous les bytes sauf le premier (header 0xFB)
    # Le checksum ne s'inclut pas lui-même dans le calcul
    checksum = sum(data[1:]) & 0xFF
    return checksum



def start_udp(port=dest_port):
    """Démarre le socket UDP"""
    global sock, dest_port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.setblocking(False)  # Mode non-bloquant pour asyncio
        dest_port = port
        print(f" UDP Socket démarré sur 0.0.0.0:{dest_port}")
        return True
    except Exception as e:
        print(f" Erreur démarrage UDP: {e}")
        return False
async def listen_for_clients():
    """Écoute les messages des clients pour les enregistrer"""
    global sock, connected_clients

    print(" Écoute des clients sur le port UDP...")

    while True:
        try:
            # Utiliser asyncio pour recevoir des données sans bloquer
            loop = asyncio.get_event_loop()
            data, addr = await loop.sock_recvfrom(sock, 1024)

            # Enregistrer le client s'il n'est pas déjà dans la liste
            if addr not in connected_clients:
                connected_clients.add(addr)
                print(f" Nouveau client connecté: {addr[0]}:{addr[1]}")
                print(f" Total clients: {len(connected_clients)}")

            # Si le client envoie "DISCONNECT", le retirer
            if data.decode("utf-8", errors="ignore").strip() == "DISCONNECT":
                connected_clients.discard(addr)
                print(f" Client déconnecté: {addr[0]}:{addr[1]}")
                print(f" Total clients: {len(connected_clients)}")

        except asyncio.CancelledError:
            print("\n Tâche d'écoute arrêtée")
            break
        except Exception as e:
            # Ignorer les erreurs de socket non-bloquant
            if "Resource temporarily unavailable" not in str(e):
                print(f" Erreur écoute: {e}")
            await asyncio.sleep(0.1)


def convert_and_send(json_data):
    """
    Convertit les données JSON et envoie via UDP à tous les clients connectés

    Args:
        json_data: Dictionnaire avec 'len' et 'data'

    Returns:
        bool: True si envoi réussi, False sinon
    """
    global sock, connected_clients

    if not sock:
        print(" Socket UDP non démarré.")
        return False

    if not connected_clients:
        print(" Aucun client connecté - aucun envoi effectué")
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
                f"  • Objet: CLS=0x{cls:02X}, ID=0x{id_track:02X}, X=0x{x_hex:08X}({obj['X']}), Y=0x{y_hex:08X}({obj['Y']}), Z=0x{z_hex:08X}({obj.get('Z', 0)})"
            )

        # Calcul du Checksum (somme de tous les bytes sauf header, avant d'ajouter le checksum)
        checksum = calculate_checksum(data)

        # Ajout du Checksum à la fin (1 byte)
        data.append(checksum & 0xFF)

        print(f"\n    Checksum: 0x{checksum:02X} (somme: NB_OBJ + données objets)")

        # Envoi à tous les clients connectés
        success_count = 0
        failed_clients = []

        for client_addr in list(connected_clients):
            try:
                sock.sendto(data, client_addr)
                success_count += 1
            except Exception as e:
                print(f" Erreur envoi vers {client_addr[0]}:{client_addr[1]}: {e}")
                failed_clients.append(client_addr)

        # Retirer les clients qui ont échoué
        for failed in failed_clients:
            connected_clients.discard(failed)
            print(f" Client retiré (échec envoi): {failed[0]}:{failed[1]}")

        print(f"\n Envoyé {len(data)} bytes à {success_count} client(s)")
        print(f"    Données: {' '.join(f'{b:02X}' for b in data[:20])}...")
        return True

    except Exception as e:
        print(f" Erreur envoi: {e}")
        return False


def stop_udp():
    """Ferme le socket UDP"""
    global sock
    if sock:
        sock.close()
        sock = None
        print(" UDP Socket fermé")




@app.on_event("startup")
async def startup_event():
    """Démarre le socket UDP et les tâches au démarrage de l'API"""
    global sending_task, listener_task

    start_udp(port=dest_port)

    # Démarrer la tâche d'écoute des clients
    listener_task = asyncio.create_task(listen_for_clients())
    print(" Tâche d'écoute des clients démarrée")

    # Démarrer la tâche d'envoi automatique
    #sending_task = asyncio.create_task(auto_send_task())
    #print(" Tâche d'envoi automatique démarrée (toutes les secondes)")


@app.on_event("shutdown")
async def shutdown_event():
    """Ferme le socket UDP et arrête les tâches à l'arrêt de l'API"""
    global sending_task, listener_task

    if listener_task:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    if sending_task:
        sending_task.cancel()
        try:
            await sending_task
        except asyncio.CancelledError:
            pass

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
        print(f"{'='*50}")
    else:
        print(f"Focus already above threshold: {hybrid_score:.2f}")
    
    
    cap.release()
    cv2.destroyAllWindows()


model_path = "yolov8n.pt"
model_yolo = YOLO(model_path).to("cuda")





def track_objects_yolo(model, frame):
    results = model.track(frame, conf=0.3, persist=True, verbose=False)
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
        conf = float(box.conf[0])

        final_output.append({
            "id": str(track_id),
            "class_name": cls,
            "bbox": bbox,
            "conf": conf
        })

    return final_output








process_thread = None
stop_flag = False


def string_to_hex(s):
    return s.encode("utf-8").hex()

def process_data(data):
    list_of = []
    for obj in data:
        x1 , y1, x2, y2  = obj["bbox"]
        X , Y  = int((x1*2+x2*2)/2), int((y1*2 + y2*2)/2)
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




def IA_process(rtsp_RGB = RTSP_RGP, rtsp_thermique = RTSP_THER, track_all = True, ids_to_track = []):
    print("start process ... ")
    global stop_flag, memo, backup, track_only_id

    track_only_id = None 

    
    cap_rgb = cv2.VideoCapture(rtsp_RGB)
    cap_ther = cv2.VideoCapture(rtsp_thermique)

    last_detection_time = time.time()

    while True and not stop_flag:
        # try:
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_ther, frame_ther = cap_ther.read()

            
            if not ret_rgb and  not ret_ther:
                break
            frame_rgbc = frame_rgb.copy()
            frame_thc = frame_ther.copy()

            start_time = time.time()
            if last_detection_time and (time.time() - last_detection_time) > 10:
                last_detection_time = time.time()
                print("skipping frame due to no detections...")
                continue


            frame_rgb = cv2.resize(frame_rgb, (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2))
            frame_ther = cv2.resize(frame_ther, (frame_ther.shape[1] // 2, frame_ther.shape[0] // 2))

            update_buffer(frame_rgbc, BUFFER)
            update_buffer(frame_thc,BUFFER_t)
            
            yolo_results = track_objects_yolo(model_yolo, frame_rgb)

            final_results = yolo_results


            if track_only_id is not None:

                final_results = [obj for obj in final_results if string_to_hex(obj["id"]) == track_only_id]
                memo = {track_only_id: memo.get(track_only_id, [])}


            if len(final_results) > 0 :

                
                
                data_event = process_data(final_results)

                STATIC_DATA = {"len" : len(data_event) , "data" : data_event}
                print("STATIC_DATA  : ",STATIC_DATA)
                convert_and_send(STATIC_DATA)
                
                for object in final_results:
                    track_id_t = object["id"]
                    cls_name_t = object["class_name"]

                    box = list(map(int, object["bbox"]))
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2

                    x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2

                    track_key = string_to_hex(track_id_t)

                    if track_key not in memo:
                        memo[track_key] = []
                    memo[track_key].append((x_c, y_c))

                    if len(memo[track_key]) > 50:
                        memo[track_key].pop(0)

                            
                    key = f"{track_id_t}-{cls_name_t}"
                    if key not in BUFFER_obj:
                        BUFFER_obj[key] = [deque(maxlen=150),deque(maxlen=150)]

                    cropped_rgb = crop(frame_rgbc,[x1, y1, x2, y2])
                    cropped_ther = crop(frame_thc,[x1, y1, x2, y2])

                    update_buffer(cropped_rgb, BUFFER_obj[key][0])
                    update_buffer(cropped_ther,BUFFER_obj[key][1])

                    save_data(frame_rgbc, frame_thc, object, BUFFER,BUFFER_t, BUFFER_obj[key])


            End_time = time.time()
            elapsed = End_time - start_time
            print(elapsed)
        # except Exception as e:
        #     print(f"Error processing frame: {e}, reinitializing VideoCapture...")
        #     cap_rgb.release()
        #     cap_ther.release()
        #     cap_rgb = cv2.VideoCapture(rtsp_RGB)
        #     cap_ther = cv2.VideoCapture(rtsp_thermique)
        #     continue  


            cap_rgb.release()
            cap_ther.release()
            cv2.destroyAllWindows()







def move_camera_to_track(target_pos):
    cap_rgb_ = cv2.VideoCapture("rtsp://admin:2899100*-+@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")
    _ , frame_ = cap_rgb_.read()
    h_rgb, w_rgb = frame_.shape[:2]
    CAM_CENTER_X = int(w_rgb/2) 
    CAM_CENTER_Y = int(h_rgb/2)
    
    THRESH_X = 50      
    THRESH_Y = 50
    x, y = target_pos
    dx = x - CAM_CENTER_X
    dy = y - CAM_CENTER_Y

    if abs(dx) > THRESH_X:
        direction_x = "right" if dx > 0 else "left"
        speed_x = min(abs(dx)//10, 8)
        send_ptz_request(direction_x, speed_x)
        time.sleep(0.5)
        send_ptz_request_stop(direction_x, speed_x)

    if abs(dy) > THRESH_Y:
        direction_y = "down" if dy > 0 else "up"
        speed_y = min(abs(dy)//10, 8)
        send_ptz_request(direction_y, speed_y)
        time.sleep(0.5)
        send_ptz_request_stop(direction_y, speed_y)


def send_ptz_request(direction, speed=4):
    url = f"{FOCUS_SERVER}/ptz/cam1/move"
    payload = {
        "direction": direction,
        "speed": speed,

    }
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code != 200:
            print(f"PTZ request failed: {r.status_code}")
        else:
            print(f"PTZ request sent successfully: {direction}")
    except Exception as e:
        print(f"PTZ request error: {e}")

def send_ptz_request_stop(direction, speed=8):
    url = f"{FOCUS_SERVER}/ptz/cam2/stop"
    payload = {
        "direction": direction,
        "speed": speed
    }
    try:
        r = requests.post(url, json=payload, timeout=0.5)
        if r.status_code != 200:
            print(f"PTZ request failed: {r.status_code}")
        else:
            print(f"PTZ request sent successfully: {direction}")
    except Exception as e:
        print(f"PTZ request error: {e}")

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

@app.post("/track/object/{id}")
async def track_object(id: str):
    global memo, track_only_id
    
    target_hex =id
    track_only_id = target_hex 
    
    if target_hex in memo:
        positions = memo[target_hex]
        last_pos = positions[-1] if positions else None

        if last_pos:
            move_camera_to_track(last_pos)

        return {
            "status": "found",
            "id": id,
            "hex_id": target_hex,
            "last_position": last_pos,
            "path_length": len(positions)
        }
    
    last_status = backup[-1] if backup else None
    
    return {
        "status": "not found",
        "id": id,
        "last_status": last_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9898
    )

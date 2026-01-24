"""Camera processing module for video capture and object detection."""
import cv2
import time
import json
import threading
import collections
from threading import Lock
from typing import Dict, List
from datetime import datetime
from ultralytics import YOLO

from .config import *
from .utils import draw_bbox_with_label, box_center
from .tracker import assign_circular_id
from .ptz_control import ptz_follow
from .storage import save_data
from .udp_handler import convert_and_send


def load_coco_classes(json_path):
    """Load COCO class names from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


COCO_CLASSES = load_coco_classes(CLASSES_JSON_PATH)


class CameraProcessor:
    """Process video frames for a single camera."""
    
    def __init__(self, cam_id, video_file, model_path, confidence, queue_size=300):
        self.cam_id = cam_id
        self.video_file = video_file
        self.model_path = model_path
        self.confidence = confidence
        
        # Queues
        self.frame_queue = collections.deque(maxlen=queue_size)
        self.frame_queue_lock = Lock()
        self.webrtc_queue = collections.deque(maxlen=WEBRTC_QUEUE_SIZE)
        self.webrtc_queue_lock = Lock()
        
        # Control
        self.stop_event = threading.Event()
        self.cap_thread = None
        self.proc_thread = None
        
        # Tracking
        self.tracked_ids: Dict[int, List[List[int]]] = {}
        self.target_id = None
        self.track_enabled = False
        
        # Model
        self.model = None
        
        # Frame timing
        self.last_input_timestamp = None
        self.input_frame_interval = None
        self.camera_fps = None
        
        # Display
        self.frame_show = None
        self.orig_h = 0
        self.orig_w = 0
        
        # Last send time for UDP
        self.last_send_time = 0
        
    def start(self):
        """Start camera capture and processing threads."""
        self.stop_event.clear()
        
        self.model = YOLO(self.model_path)
        self.model.to("cuda")
        
        if self.cap_thread is None:
            self.cap_thread = threading.Thread(target=self._frame_reader, daemon=True)
            self.cap_thread.start()
        
        if self.proc_thread is None:
            self.proc_thread = threading.Thread(target=self._frame_processor, daemon=True)
            self.proc_thread.start()
    
    def stop(self):
        """Stop camera capture and processing."""
        self.stop_event.set()
        self.track_enabled = False
        self.target_id = None
        
        if self.cap_thread:
            self.cap_thread.join(timeout=1)
            self.cap_thread = None
        
        if self.proc_thread:
            self.proc_thread.join(timeout=1)
            self.proc_thread = None
        
        with self.frame_queue_lock:
            self.frame_queue.clear()
        
        self.model = None
        self.frame_show = None
    
    def _frame_reader(self):
        """Read frames from video source."""
        cap = cv2.VideoCapture(self.video_file)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"❌ Video file not opened: {self.video_file}")
            self.stop_event.set()
            return
        
        # Get actual FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.camera_fps = fps
        else:
            self.camera_fps = 25.0
        print(f"📹 {self.cam_id.upper()} video FPS: {self.camera_fps}")
        
        print(f"✅ Frame reader {self.cam_id.upper()} started")
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            timestamp = time.time()
            with self.frame_queue_lock:
                self.frame_queue.append((frame, timestamp))
        
        cap.release()
        print(f"🛑 Frame reader {self.cam_id.upper()} stopped")
    
    def _frame_processor(self):
        """Process frames with YOLO detection."""
        print(f"✅ Processing thread {self.cam_id.upper()} started")
        
        while not self.stop_event.is_set():
            # Check if frames are available
            with self.frame_queue_lock:
                if len(self.frame_queue) == 0:
                    time.sleep(0.01)
                    continue
                # Peek at frame without removing it yet
                frame, frame_timestamp = self.frame_queue[0]
            
            # Calculate actual input frame interval
            if self.last_input_timestamp is not None:
                self.input_frame_interval = frame_timestamp - self.last_input_timestamp
            
            try:
                self.orig_h, self.orig_w = frame.shape[:2]
                scale_x, scale_y = int(self.orig_h / TARGET_HEIGHT), int(self.orig_w / TARGET_WIDTH)
                framec = frame.copy()
                frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                
                # Initialize frame_show if needed
                if self.frame_show is None:
                    self.frame_show = frame_resized.copy()
                
                h, w = frame_resized.shape[:2]
                frame_cx, frame_cy = w // 2, h // 2
                
                final_output = []
                current_frame_ids = set()
                
                # Start with clean frame for this iteration
                output_frame = frame_resized.copy()
                
                results = self.model.track(
                    framec,
                    persist=True,
                    conf=self.confidence,
                    verbose=False
                )
                
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for i, track_id in enumerate(track_ids):
                        cls = int(results[0].boxes.cls[i].item())
                        conf = float(results[0].boxes.conf[i].item())
                        circ_id = int(assign_circular_id(track_id, self.cam_id))
                        
                        x1, y1, x2, y2 = map(int, boxes[i])
                        
                        if self.target_id is not None and circ_id != self.target_id:
                            continue
                        
                        current_frame_ids.add(circ_id)
                        
                        final_output.append({
                            "id": circ_id,
                            "hex_id": f"{circ_id:02x}",
                            "class_name": cls,
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "timestamp": frame_timestamp,
                        })
                        
                        label_text = f"{COCO_CLASSES.get(cls, 'unkn')} ID:{circ_id} {conf:.2f}"
                        cx, cy = box_center(x1, y1, x2, y2)
                        box_color = SIMPLE_ID_TO_COLOR.get(cls, (0, 180, 255))
                        text_color = (255, 255, 255)
                        
                        if self.track_enabled and self.target_id is not None:
                            framec = ptz_follow(self.cam_id, circ_id, cx, cy,
                                              framec, frame_cx, frame_cy, self.target_id)
                        
                        output_frame = draw_bbox_with_label(
                            framec, (x1, y1, x2, y2), label_text, box_color, text_color
                        )
                        output_frame = cv2.resize(output_frame, (TARGET_WIDTH, TARGET_HEIGHT))
                        
                        self.tracked_ids.setdefault(circ_id, []).append((cx, cy))
                        self.tracked_ids[circ_id] = self.tracked_ids[circ_id][-50:]
                    
                    # Clean up lost tracks
                    for cid in list(self.tracked_ids.keys()):
                        if cid not in current_frame_ids:
                            del self.tracked_ids[cid]
                    
                    # Send UDP data
                    if len(final_output) > 0:
                        current_time = time.time()
                        if current_time - self.last_send_time >= 1.0:
                            data_event = self._process_data(final_output)
                            STATIC_DATA = {"len": len(data_event), "data": data_event}
                            convert_and_send(STATIC_DATA)
                            self.last_send_time = current_time
                        
                        for obj in final_output:
                            save_data(framec, self.cam_id, obj, scale_x, scale_y, COCO_CLASSES)
                
                # Update frame_show for display
                self.frame_show = output_frame
                
                # ALWAYS push every frame to WebRTC queue - no frames are lost
                with self.webrtc_queue_lock:
                    self.webrtc_queue.append(output_frame.copy())
                
                # Only remove frame from queue after successful processing
                with self.frame_queue_lock:
                    if len(self.frame_queue) > 0:
                        self.frame_queue.popleft()
                
                # Update timestamp only after successful processing
                self.last_input_timestamp = frame_timestamp
            
            except Exception as e:
                print(f"❌ Error in processing {self.cam_id} frame: {e}")
                # Remove the problematic frame to avoid infinite loop
                with self.frame_queue_lock:
                    if len(self.frame_queue) > 0:
                        self.frame_queue.popleft()
                continue
        
        print(f"🛑 Processing thread {self.cam_id.upper()} stopped")
    
    def _process_data(self, data):
        """Convert detection data to UDP format."""
        list_of = []
        for obj in data:
            x1, y1, x2, y2 = obj["bbox"]
            X1, Y1, X2, Y2 = int(x1), int(y1), int(x2), int(y2)
            ID_TRACK = hex(obj["id"])
            CLS = hex(obj["class_name"])
            list_of.append({
                "CLS": CLS,
                "ID_TRACK": ID_TRACK,
                "X1": X1,
                "Y1": Y1,
                "X2": X2,
                "Y2": Y2,
                "Z": 0,
            })
        return list_of

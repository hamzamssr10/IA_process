"""Configuration settings for the AI surveillance system."""

# Video sources
RTSP_URL_1 = "rtsp://localhost:8554/cam2"
RTSP_URL_2 = "rtsp://localhost:8554/cam1"
VIDEO_FILE_1 = RTSP_URL_1
VIDEO_FILE_2 = RTSP_URL_2

# Video processing
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Model paths
MODEL_PATH_CAM1 = "models/best_mm.pt"
MODEL_PATH_CAM2 = "/home/ubuntu/IA_process/best_yarb.pt"
CLASSES_JSON_PATH = "data/therm_classes.json"

# Queue sizes
FRAME_QUEUE_SIZE = 300
WEBRTC_QUEUE_SIZE = 40

# Detection confidence thresholds
CONFIDENCE_CAM1 = 0.9
CONFIDENCE_CAM2 = 0.4

# UDP settings
UDP_PORT = 52383

# Focus server
FOCUS_SERVER = "http://localhost:3000"

# Storage
SAVE_BASE = "/home/ubuntu/falcon_camera_udp_workers/stockage/ftp_storage/IA/"

# Video recording
VIDEO_SAVE_DIR = "data/recordings"
VIDEO_RECORD_DURATION = 10  # seconds

# State file
STATE_FILE = "/home/ubuntu/IA_PROCESS/app_state.json"

# Class colors (simple_id -> BGR color)
SIMPLE_ID_TO_COLOR = {
    0: (0, 255, 0),    # Green
    1: (255, 0, 0),    # Blue
    2: (0, 0, 255)     # Red
}

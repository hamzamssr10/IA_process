"""Storage module for saving detection events."""
import os
import json
import cv2
from datetime import datetime
from threading import Lock
from .config import SAVE_BASE

os.makedirs(SAVE_BASE, exist_ok=True)

saved_objects = set()
current_day_hour = None
lock = Lock()


def save_data(full_frame, cam, obj, fx, fy, COCO_CLASSES, event_type="event"):
    """Save detection data including frames and metadata."""
    global saved_objects, current_day_hour
    
    with lock:
        now = datetime.now()
        half_hour = 0 if now.minute < 30 else 30
        ts_ = now.strftime(f"%Y%m%d_%H{half_hour:02d}")
        ts = now.strftime("%Y%m%d_%H%M%S")

        # Reset saved objects if half-hour changed
        if current_day_hour != ts_:
            saved_objects.clear()
            current_day_hour = ts_

        obj_id = obj["id"]

        # Skip if already saved in this time slot
        if obj_id in saved_objects:
            return
        saved_objects.add(obj_id)

        class_name = obj["class_name"]
        cls_name = COCO_CLASSES.get(class_name, "unkn")
        conf = obj["conf"]
        x1, y1, x2, y2 = map(int, obj["bbox"])
        x1, y1, x2, y2 = int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy)

        # Extract crop
        crop_1 = full_frame[y1:y2, x1:x2]

        # Build directory
        base_dir = os.path.join(SAVE_BASE, now.strftime("%Y-%m-%d"))
        os.makedirs(base_dir, exist_ok=True)

        prefix = f"{cls_name}-{ts}-{conf:.2f}"

        if cam == "cam1":
            # Save RGB camera data
            full_path_1 = os.path.join(base_dir, f"{prefix}-full_rgb.jpg")
            if full_frame.size > 0:
                cv2.imwrite(full_path_1, full_frame)
            else:
                full_path_1 = None

            crop_path_1 = os.path.join(base_dir, f"{prefix}-crop_rgb.jpg")
            if crop_1.size > 0:
                cv2.imwrite(crop_path_1, crop_1)
            else:
                crop_path_1 = None

            meta_path = os.path.join(base_dir, f"{prefix}-meta.json")
            metadata = {
                "timestamp": ts,
                "event": event_type,
                "class": class_name,
                "confidence": conf,
                "bbox": obj["bbox"],
                "full_rgb": full_path_1,
                "crop_rgb": crop_path_1,
            }

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)

        else:
            # Save thermal camera data
            full_path_1 = os.path.join(base_dir, f"{prefix}-full_ther.jpg")
            if full_frame.size > 0:
                cv2.imwrite(full_path_1, full_frame)
            else:
                full_path_1 = None

            crop_path_1 = os.path.join(base_dir, f"{prefix}-crop_ther.jpg")
            if crop_1.size > 0:
                cv2.imwrite(crop_path_1, crop_1)
            else:
                crop_path_1 = None

            meta_path = os.path.join(base_dir, f"{prefix}-meta.json")
            metadata = {
                "timestamp": ts,
                "event": event_type,
                "class": class_name,
                "confidence": conf,
                "bbox": obj["bbox"],
                "full_ther": full_path_1,
                "crop_ther": crop_path_1,
            }

            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"[SAVE] Saved to → {base_dir}")

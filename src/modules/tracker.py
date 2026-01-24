"""Tracking ID management module."""
from threading import Lock

tracker_state = {
    "cam1": {
        "map": {},
        "current_id": 0
    },
    "cam2": {
        "map": {},
        "current_id": 0
    }
}

tracker_lock = Lock()


def assign_circular_id(real_id: int, cam: str) -> int:
    """
    Convert YOLO tracking ID to a circular ID (0–255), per camera.
    Same object keeps its circular ID for that camera only.
    """
    if cam not in tracker_state:
        raise ValueError(f"Unknown camera '{cam}'")

    with tracker_lock:
        cam_map = tracker_state[cam]["map"]
        current_id = tracker_state[cam]["current_id"]

        if real_id not in cam_map:
            cam_map[real_id] = current_id
            tracker_state[cam]["current_id"] = (current_id + 1) % 256

        return cam_map[real_id]


def coco_id_to_simple_id(coco_id):
    """Map COCO class IDs to simplified categories."""
    vehicle_ids = [1, 2, 3, 4, 5, 6, 7]
    person_ids = [0]
    animal_ids = [15, 16, 17, 18, 19, 20, 21, 22, 23]

    if coco_id in vehicle_ids:
        return 2
    elif coco_id in person_ids:
        return 0
    elif coco_id in animal_ids:
        return 1
    else:
        return None

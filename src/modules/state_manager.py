"""State management for tracking camera status."""
import os
import json
from .config import STATE_FILE


def read_state():
    """Read application state from file."""
    if not os.path.exists(STATE_FILE):
        return {
            "cam1": {"tracking": "stopped", "follow": "stopped", "focus": "stopped"},
            "cam2": {"tracking": "stopped", "follow": "stopped", "focus": "stopped"}
        }
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def write_state(state: dict):
    """Write application state to file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

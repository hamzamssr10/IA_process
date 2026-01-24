"""Main application entry point - Refactored modular version."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription

from modules.config import VIDEO_FILE_1, VIDEO_FILE_2, MODEL_PATH_CAM1, MODEL_PATH_CAM2, CONFIDENCE_CAM1, CONFIDENCE_CAM2
from modules.camera_processor import CameraProcessor
from modules.webrtc_handler import CameraTrack
from modules.state_manager import read_state, write_state
from modules.focus import monitor_focus
from modules.udp_handler import start_udp, stop_udp, listen_for_clients_thread

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global variables
pcs = set()
listener_thread = None
focus_sessions = {}

# Camera processors
camera_processors = {
    "cam1": CameraProcessor("cam1", VIDEO_FILE_1, MODEL_PATH_CAM1, CONFIDENCE_CAM1),
    "cam2": CameraProcessor("cam2", VIDEO_FILE_2, MODEL_PATH_CAM2, CONFIDENCE_CAM2)
}


@app.on_event("startup")
async def startup_event():
    """Start UDP and restore previous state."""
    global listener_thread
    
    start_udp()
    
    # Start UDP listener thread
    listener_thread = threading.Thread(target=listen_for_clients_thread, daemon=True)
    listener_thread.start()
    print("🎧 UDP listener thread started")
    
    state = read_state()
    print("✅ App started with state:", state)
    
    # Restore CAM1 state
    if state["cam1"]["tracking"] == "running":
        camera_processors["cam1"].start()
        print("🔄 CAM1 tracking resumed")
    
    if state["cam1"]["follow"] == "running":
        state["cam1"]["follow"] = "stopped"
        write_state(state)
    
    if state["cam1"]["focus"] == "running":
        stop_event = threading.Event()
        focus_sessions["cam1"] = stop_event
        threading.Thread(target=monitor_focus, args=("cam1", VIDEO_FILE_1, stop_event), daemon=True).start()
        print("🔄 CAM1 focus resumed")
    
    # Restore CAM2 state
    if state["cam2"]["tracking"] == "running":
        camera_processors["cam2"].start()
        print("🔄 CAM2 tracking resumed")
    
    if state["cam2"]["follow"] == "running":
        state["cam2"]["follow"] = "stopped"
        write_state(state)
    
    if state["cam2"]["focus"] == "running":
        stop_event = threading.Event()
        focus_sessions["cam2"] = stop_event
        threading.Thread(target=monitor_focus, args=("cam2", VIDEO_FILE_2, stop_event), daemon=True).start()
        print("🔄 CAM2 focus resumed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global listener_thread
    
    if listener_thread and listener_thread.is_alive():
        listener_thread.join(timeout=5)
    
    stop_udp()
    
    for pc in pcs:
        await pc.close()
    pcs.clear()


# WebRTC endpoints
@app.post("/{cam}")
async def webrtc_offer(cam: str, request: Request):
    """Handle WebRTC offer."""
    try:
        sdp = await request.body()
        sdp = sdp.decode("utf-8")
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        
        print(f"\n{'='*50}")
        print(f"📥 Received offer from client for {cam}")
        print(f"{'='*50}")
        
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"🔗 Connection state: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                pcs.discard(pc)
        
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"🧊 ICE connection state: {pc.iceConnectionState}")
        
        await pc.setRemoteDescription(offer)
        print("✅ Remote description set")
        
        # Add video track
        video_track = CameraTrack(camera_processors[cam])
        pc.addTrack(video_track)
        print("✅ Video track added")
        
        # Configure transceivers
        for transceiver in pc.getTransceivers():
            if transceiver.direction is None:
                transceiver.direction = "recvonly"
            if transceiver.kind == "audio":
                transceiver.direction = "inactive"
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        print("✅ Answer created and set")
        print(f"{'='*50}\n")
        
        return JSONResponse({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    
    except Exception as e:
        print(f"❌ Error in webrtc_offer: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# CAM1 Tracking endpoints
@app.post("/ia_process/trackobject/cam1/start")
async def start_tracking_cam1():
    """Start object tracking for camera 1."""
    camera_processors["cam1"].start()
    
    state = read_state()
    state["cam1"]["tracking"] = "running"
    write_state(state)
    
    return {"status": "started"}


@app.post("/ia_process/trackobject/cam1/stop")
async def stop_tracking_cam1():
    """Stop object tracking for camera 1."""
    camera_processors["cam1"].stop()
    
    state = read_state()
    state["cam1"]["tracking"] = "stopped"
    write_state(state)
    
    return {"status": "stopped"}


@app.post("/ia_process/trackobject_ids/cam1/start/{ids}")
async def follow_object_cam1(ids: int):
    """Start following specific object on camera 1."""
    camera_processors["cam1"].target_id = ids
    camera_processors["cam1"].track_enabled = True
    
    state = read_state()
    state["cam1"]["follow"] = "running"
    write_state(state)
    
    return {"status": "following", "id": ids}


@app.post("/ia_process/trackobject_ids/cam1/stop")
async def stop_follow_cam1():
    """Stop following object on camera 1."""
    camera_processors["cam1"].track_enabled = False
    camera_processors["cam1"].target_id = None
    
    state = read_state()
    state["cam1"]["follow"] = "stopped"
    write_state(state)
    
    return {"status": "follow stopped"}


# CAM1 Focus endpoints
@app.post("/ia_process/focus/cam1/start")
def start_focus_cam1():
    """Start auto-focus for camera 1."""
    global focus_sessions
    
    if "cam1" in focus_sessions:
        return {"status": "already_running", "camera": "cam1"}
    
    stop_event = threading.Event()
    focus_sessions["cam1"] = stop_event
    
    threading.Thread(
        target=monitor_focus,
        args=("cam1", VIDEO_FILE_1, stop_event),
        daemon=True
    ).start()
    
    state = read_state()
    state["cam1"]["focus"] = "running"
    write_state(state)
    
    return {"status": "started", "camera": "cam1", "action": "autofocus"}


@app.post("/ia_process/focus/cam1/stop")
def stop_focus_cam1():
    """Stop auto-focus for camera 1."""
    stop_event = focus_sessions.get("cam1")
    
    state = read_state()
    state["cam1"]["focus"] = "stopped"
    write_state(state)
    
    if stop_event is None:
        return {"status": "not_running", "camera": "cam1"}
    
    stop_event.set()
    focus_sessions.pop("cam1", None)
    
    return {"status": "stopped", "camera": "cam1", "action": "autofocus"}


# CAM2 Tracking endpoints
@app.post("/ia_process/trackobject/cam2/start")
async def start_tracking_cam2():
    """Start object tracking for camera 2."""
    camera_processors["cam2"].start()
    
    state = read_state()
    state["cam2"]["tracking"] = "running"
    write_state(state)
    
    return {"status": "started"}


@app.post("/ia_process/trackobject/cam2/stop")
async def stop_tracking_cam2():
    """Stop object tracking for camera 2."""
    camera_processors["cam2"].stop()
    
    state = read_state()
    state["cam2"]["tracking"] = "stopped"
    write_state(state)
    
    return {"status": "stopped"}


@app.post("/ia_process/trackobject_ids/cam2/start/{ids}")
async def follow_object_cam2(ids: int):
    """Start following specific object on camera 2."""
    camera_processors["cam2"].target_id = ids
    camera_processors["cam2"].track_enabled = True
    
    state = read_state()
    state["cam2"]["follow"] = "running"
    write_state(state)
    
    return {"status": "following", "id": ids}


@app.post("/ia_process/trackobject_ids/cam2/stop")
async def stop_follow_cam2():
    """Stop following object on camera 2."""
    camera_processors["cam2"].track_enabled = False
    camera_processors["cam2"].target_id = None
    
    state = read_state()
    state["cam2"]["follow"] = "stopped"
    write_state(state)
    
    return {"status": "follow stopped"}


# CAM2 Focus endpoints
@app.post("/ia_process/focus/cam2/start")
def start_focus_cam2():
    """Start auto-focus for camera 2."""
    global focus_sessions
    
    if "cam2" in focus_sessions:
        return {"status": "already_running", "camera": "cam2"}
    
    stop_event = threading.Event()
    focus_sessions["cam2"] = stop_event
    
    threading.Thread(
        target=monitor_focus,
        args=("cam2", VIDEO_FILE_2, stop_event),
        daemon=True
    ).start()
    
    state = read_state()
    state["cam2"]["focus"] = "running"
    write_state(state)
    
    return {"status": "started", "camera": "cam2", "action": "autofocus"}


@app.post("/ia_process/focus/cam2/stop")
def stop_focus_cam2():
    """Stop auto-focus for camera 2."""
    stop_event = focus_sessions.get("cam2")
    
    state = read_state()
    state["cam2"]["focus"] = "stopped"
    write_state(state)
    
    if stop_event is None:
        return {"status": "not_running", "camera": "cam2"}
    
    stop_event.set()
    focus_sessions.pop("cam2", None)
    
    return {"status": "stopped", "camera": "cam2", "action": "autofocus"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9898)

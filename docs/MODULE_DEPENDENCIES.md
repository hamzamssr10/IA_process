# Module Dependency Graph

## Dependency Structure

```
main_refactored.py
├── config.py (no dependencies)
├── state_manager.py
│   └── config.py (STATE_FILE)
├── udp_handler.py (no external dependencies)
├── tracker.py (no external dependencies)
├── focus.py
│   ├── config.py (FOCUS_SERVER)
│   └── ptz_control.py
├── ptz_control.py
│   └── config.py (FOCUS_SERVER)
├── utils.py (no external dependencies)
├── storage.py
│   └── config.py (SAVE_BASE)
├── camera_processor.py
│   ├── config.py (all constants)
│   ├── utils.py (draw_bbox_with_label, box_center)
│   ├── tracker.py (assign_circular_id)
│   ├── ptz_control.py (ptz_follow)
│   ├── storage.py (save_data)
│   └── udp_handler.py (convert_and_send)
└── webrtc_handler.py
    └── camera_processor.py (CameraProcessor)
```

## Import Order (Bottom-Up)

1. **Level 0 (No Dependencies)**
   - `config.py`
   - `udp_handler.py`
   - `tracker.py`
   - `utils.py`

2. **Level 1 (Depends on Level 0)**
   - `state_manager.py` (uses config.py)
   - `ptz_control.py` (uses config.py)
   - `storage.py` (uses config.py)

3. **Level 2 (Depends on Level 0-1)**
   - `focus.py` (uses config.py, ptz_control.py)
   - `camera_processor.py` (uses config.py, utils.py, tracker.py, ptz_control.py, storage.py, udp_handler.py)

4. **Level 3 (Depends on Level 0-2)**
   - `webrtc_handler.py` (uses camera_processor.py)

5. **Level 4 (Application Entry)**
   - `main_refactored.py` (uses all modules)

## Circular Dependency Prevention

The modular structure avoids circular dependencies:
- **config.py** is the foundation with no imports
- **utils.py**, **udp_handler.py**, **tracker.py** have no internal dependencies
- **camera_processor.py** imports from lower levels only
- **webrtc_handler.py** only depends on camera_processor
- **main_refactored.py** sits at the top and orchestrates everything

## Shared State

### Global Variables in camera_processor.py:
- `FRAME_QUEUE_1/2` - Ring buffers for raw frames (maxlen=300)
- `WEBRTC_QUEUE_CAM1/2` - Output queues for WebRTC (maxlen=500)
- `LAST_INPUT_TIMESTAMP_1/2` - Track previous frame timestamp
- `INPUT_FRAME_INTERVAL_1/2` - Measured time between frames
- `CAMERA_FPS_1/2` - Detected FPS from video source
- Locks for thread-safe queue access

### Global Variables in udp_handler.py:
- `sock` - UDP socket
- `dest_port` - UDP port
- `connected_clients` - Set of connected client addresses
- `clients_lock` - Thread lock for client list
- `stop_listener_flag` - Flag to stop listener thread

### Global Variables in tracker.py:
- `tracker_state` - Dictionary mapping (real_id, cam) to circular_id
- `tracker_lock` - Thread lock for tracker state

### Global Variables in main_refactored.py:
- `pcs` - Set of RTCPeerConnection objects
- `listener_thread` - UDP listener thread
- `focus_sessions` - Dictionary of active focus sessions
- `camera_processors` - Dictionary of CameraProcessor instances

## Thread Architecture

```
Main Thread (FastAPI/Uvicorn)
├── UDP Listener Thread (listen_for_clients_thread)
├── CAM1 Threads
│   ├── Frame Reader Thread (_frame_reader)
│   ├── Frame Processor Thread (_frame_processor)
│   └── Focus Monitor Thread (monitor_focus) [optional]
└── CAM2 Threads
    ├── Frame Reader Thread (_frame_reader)
    ├── Frame Processor Thread (_frame_processor)
    └── Focus Monitor Thread (monitor_focus) [optional]

WebRTC Async Tasks (aiortc event loop)
├── CameraTrack.recv() - CAM1
└── CameraTrack.recv() - CAM2
```

## Data Flow

```
Video File
    ↓
[Frame Reader Thread]
    ↓ (frame, timestamp)
FRAME_QUEUE (deque, maxlen=300)
    ↓
[Frame Processor Thread]
    ├→ YOLO Detection
    ├→ Circular ID Assignment
    ├→ PTZ Following (if enabled)
    ├→ UDP Data Sending
    ├→ Event Storage
    └→ Draw Bounding Boxes
        ↓ (processed frame)
WEBRTC_QUEUE (deque, maxlen=500)
    ↓
[WebRTC Track]
    ↓ (timed delivery based on INPUT_FRAME_INTERVAL)
Browser Client
```

## Key Timing Mechanism

```
Frame Reader:
    timestamp = time.time()
    queue.append((frame, timestamp))

Frame Processor:
    frame, frame_timestamp = queue.popleft()
    if last_timestamp is not None:
        INPUT_FRAME_INTERVAL = frame_timestamp - last_timestamp
    last_timestamp = frame_timestamp

WebRTC Track:
    actual_interval = camera_processor.input_frame_interval
    if last_frame_time is not None:
        sleep_time = actual_interval - (time.time() - last_frame_time)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
```

This ensures WebRTC output timing matches actual input video timing, not calculated FPS.

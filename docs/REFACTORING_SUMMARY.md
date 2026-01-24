# AI Surveillance System - Refactoring Summary

## Overview
The monolithic `main.py` (~1800 lines) has been successfully refactored into a modular architecture with 11 separate modules for better maintainability, testability, and code organization.

## Module Structure

### 1. **config.py** (Configuration)
- **Purpose**: Centralized configuration constants
- **Contents**:
  - Video source paths (VIDEO_FILE_1/2, RTSP_URL_1/2)
  - Model paths and confidence thresholds
  - Queue sizes (FRAME_QUEUE_SIZE, WEBRTC_QUEUE_SIZE)
  - Target dimensions (TARGET_WIDTH, TARGET_HEIGHT)
  - UDP settings
  - Focus server URL
  - Storage paths
  - State file location
  - Color mappings for classes

### 2. **udp_handler.py** (UDP Communication)
- **Purpose**: Manage UDP socket communication for sending detection data
- **Key Functions**:
  - `start_udp(port)` - Initialize UDP socket
  - `stop_udp()` - Close UDP socket
  - `listen_for_clients_thread()` - Listen for client connections
  - `convert_and_send(json_data)` - Convert and send detection data to all connected clients
  - `calculate_checksum(data)` - Calculate packet checksums

### 3. **utils.py** (Drawing Utilities)
- **Purpose**: Image processing and drawing utilities
- **Key Functions**:
  - `draw_bbox_with_label()` - Draw bounding boxes with transparent rounded labels
  - `box_center()` - Calculate center point of bounding box
- **Features**: Dynamic font sizing, rounded corners, transparency blending

### 4. **state_manager.py** (State Persistence)
- **Purpose**: Save and restore application state across restarts
- **Key Functions**:
  - `read_state()` - Load state from JSON file
  - `write_state(data)` - Save state to JSON file
- **State Tracked**: tracking status, follow status, focus status for each camera

### 5. **tracker.py** (Tracking ID Management)
- **Purpose**: Manage circular tracking IDs (0-255) for detected objects
- **Key Functions**:
  - `assign_circular_id(real_id, cam)` - Map YOLO tracking IDs to circular IDs
- **Features**: Maintains consistent IDs across frames, per-camera tracking

### 6. **focus.py** (Auto-Focus Module)
- **Purpose**: Implement auto-focus algorithms for camera control
- **Key Functions**:
  - `calculate_tenengrad()` - Tenengrad focus measure
  - `calculate_laplacian()` - Laplacian variance focus measure
  - `calculate_hybrid_focus()` - Combined focus score
  - `frame_reader()` - Read frames for focus analysis
  - `focus_controller()` - Main focus control loop
  - `monitor_focus()` - Monitor focus quality and adjust
  - `increase_focus()` / `decrease_focus()` - Focus adjustment commands

### 7. **ptz_control.py** (PTZ Camera Control)
- **Purpose**: Control Pan-Tilt-Zoom camera movements
- **Key Functions**:
  - `send_ptz_request()` - Send PTZ movement commands
  - `send_ptz_request_stop()` - Stop PTZ movement
  - `cam_left/right/up/down()` - Directional movement commands
  - `cam_stop()` - Stop camera movement
  - `compute_ptz_speed()` - Calculate movement speed based on error
  - `ptz_follow()` - Follow tracked object with PTZ camera

### 8. **storage.py** (Event Storage)
- **Purpose**: Save detection events, images, and metadata
- **Key Functions**:
  - `save_data()` - Save detection data including frames and metadata
- **Features**: 
  - Saves full frame and cropped detections
  - Separate handling for RGB (cam1) and thermal (cam2) cameras
  - JSON metadata with timestamps, confidence, bounding boxes
  - Deduplication within 30-minute time slots

### 9. **camera_processor.py** (Frame Processing)
- **Purpose**: Core camera processing and YOLO object detection
- **Key Class**: `CameraProcessor`
  - Manages video capture, frame queuing, YOLO detection
  - Tracks FPS from video source (not defaulting to 25)
  - Measures actual input frame intervals
  - Maintains circular tracking IDs
  - Handles PTZ following
  - Manages WebRTC output queue
- **Key Methods**:
  - `start()` - Start capture and processing threads
  - `stop()` - Stop all processing
  - `_frame_reader()` - Read frames from video source, detect FPS
  - `_frame_processor()` - Process frames with YOLO, track objects, send UDP data

### 10. **webrtc_handler.py** (WebRTC Streaming)
- **Purpose**: Stream processed video via WebRTC
- **Key Class**: `CameraTrack(VideoStreamTrack)`
  - Waits for queue to fill (500 frames) before streaming
  - Uses actual INPUT_FRAME_INTERVAL for timing (not calculated FPS)
  - Logs frame timing every 30 frames
  - Enforces frame rate based on measured input intervals
- **Key Method**:
  - `recv()` - Deliver next video frame with precise timing

### 11. **main_refactored.py** (Application Entry Point)
- **Purpose**: FastAPI application setup and API endpoint definitions
- **Components**:
  - FastAPI app with CORS middleware
  - Camera processor instances
  - WebRTC peer connection management
  - UDP listener thread management
  - Focus session management
  - State restoration on startup
- **API Endpoints**:
  - `POST /{cam}` - WebRTC offer/answer
  - `POST /ia_process/trackobject/{cam}/start` - Start tracking
  - `POST /ia_process/trackobject/{cam}/stop` - Stop tracking
  - `POST /ia_process/trackobject_ids/{cam}/start/{ids}` - Follow specific ID
  - `POST /ia_process/trackobject_ids/{cam}/stop` - Stop following
  - `POST /ia_process/focus/{cam}/start` - Start auto-focus
  - `POST /ia_process/focus/{cam}/stop` - Stop auto-focus

## Key Improvements

### 1. **FPS Detection**
- FPS is now detected from input video using `cap.get(cv2.CAP_PROP_FPS)`
- Falls back to 25 FPS only if detection fails
- Stored in `camera_fps` attribute

### 2. **WebRTC Queue Filling**
- WebRTC streaming waits until queue reaches maxlen (500 frames)
- Prevents initial buffering/startup issues
- Returns black frames while waiting

### 3. **Timestamp-Based Frame Timing**
- Measures actual time between input frames
- Stores in `input_frame_interval` attribute
- WebRTC uses measured interval instead of calculated frame duration
- Ensures output timing matches input precisely

### 4. **Code Organization**
- Each module has single responsibility
- Clear separation of concerns
- Easier to test individual components
- Improved maintainability
- Better code reusability

## Migration Guide

### To use the refactored code:

1. **Backup the old file:**
   ```powershell
   Copy-Item main.py main.py.backup
   ```

2. **Replace with refactored version:**
   ```powershell
   Copy-Item main_refactored.py main.py
   ```

3. **Verify all module files exist:**
   - config.py
   - udp_handler.py
   - utils.py
   - state_manager.py
   - tracker.py
   - focus.py
   - ptz_control.py
   - storage.py
   - camera_processor.py
   - webrtc_handler.py
   - main.py (new refactored version)

4. **Test the application:**
   ```powershell
   python main.py
   ```

## Dependencies

All dependencies remain the same:
- FastAPI
- uvicorn
- opencv-python (cv2)
- ultralytics (YOLO)
- aiortc
- numpy
- requests

## Testing Checklist

- [ ] Application starts without import errors
- [ ] UDP listener thread starts
- [ ] State restoration works on startup
- [ ] Video FPS is detected correctly
- [ ] WebRTC streams work for both cameras
- [ ] Object detection runs properly
- [ ] Tracking IDs are assigned correctly
- [ ] PTZ following works
- [ ] Auto-focus runs
- [ ] Detection data saved to storage
- [ ] UDP data sent to clients
- [ ] Frame timing matches input (check logs every 30 frames)
- [ ] WebRTC queue fills before streaming starts

## Performance Considerations

- Frame queue size: 300 frames per camera
- WebRTC queue size: 500 frames per camera (configurable in config.py)
- Threading: Separate threads for frame reading, processing, and focus monitoring
- GPU acceleration: YOLO models run on CUDA
- Frame timing: Measured intervals ensure precise output timing

## Future Enhancements

Potential areas for further improvement:
- Add unit tests for each module
- Implement configuration hot-reloading
- Add metrics/monitoring endpoints
- Implement graceful degradation on errors
- Add logging framework (replace print statements)
- Implement health check endpoints
- Add API documentation with Swagger/OpenAPI

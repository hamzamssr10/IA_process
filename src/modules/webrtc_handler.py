"""WebRTC streaming module."""
import asyncio
import time
import numpy as np
import av
from aiortc import VideoStreamTrack


class CameraTrack(VideoStreamTrack):
    """WebRTC video track for camera streaming."""
    
    def __init__(self, camera_processor):
        super().__init__()
        self.camera_processor = camera_processor
        self.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame_count = 0
        self.queue_filled = False
        self.last_frame_time = None
        
        # Use actual camera FPS
        camera_fps = camera_processor.camera_fps
        self.fps = camera_fps if camera_fps is not None else 25.0
        self.frame_duration = 1.0 / self.fps
        print(f"📺 WebRTC track {camera_processor.cam_id} initialized with FPS: {self.fps}, frame duration: {self.frame_duration:.4f}s")
    
    async def recv(self):
        """Receive next video frame."""
        # Get actual input frame interval
        actual_interval = self.camera_processor.input_frame_interval
        if actual_interval is None:
            actual_interval = self.frame_duration
        
        # Enforce frame rate timing based on actual input
        if self.last_frame_time is not None:
            elapsed = time.time() - self.last_frame_time
            sleep_time = actual_interval - elapsed
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            # Log actual time diff every 30 frames
            if self.frame_count % 30 == 0:
                actual_elapsed = time.time() - self.last_frame_time
                print(f"📊 [{self.camera_processor.cam_id}] Frame time diff: {actual_elapsed*1000:.2f}ms (input interval: {actual_interval*1000:.2f}ms, target: {self.frame_duration*1000:.2f}ms, FPS: {1/actual_elapsed:.2f})")
        
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # Wait until the queue is full before starting
        if not self.queue_filled:
            with self.camera_processor.webrtc_queue_lock:
                queue_len = len(self.camera_processor.webrtc_queue)
                queue_maxlen = self.camera_processor.webrtc_queue.maxlen
            
            if queue_len < queue_maxlen:
                print(f"⏳ Waiting for {self.camera_processor.cam_id} WEBRTC_QUEUE to fill: {queue_len}/{queue_maxlen}")
                await asyncio.sleep(0.1)
                # Return black frame while waiting
                pts, time_base = await self.next_timestamp()
                video = av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")
                video.pts = pts
                video.time_base = time_base
                return video
            else:
                self.queue_filled = True
                print(f"✅ {self.camera_processor.cam_id} WEBRTC_QUEUE is full, starting stream")
        
        pts, time_base = await self.next_timestamp()
        
        with self.camera_processor.webrtc_queue_lock:
            if len(self.camera_processor.webrtc_queue) > 0:
                frame = self.camera_processor.webrtc_queue.popleft()
                self.last_frame = frame
            else:
                # Reuse last frame to maintain FPS even if queue is empty
                frame = self.last_frame
        
        video = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video.pts = pts
        video.time_base = time_base
        return video

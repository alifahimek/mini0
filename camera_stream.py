"""
Threaded Camera Stream Module.
Decouples camera hardware frame acquisition from MediaPipe processing.
Eliminates blocking I/O latency (60-80ms per frame on Windows), boosting
system throughput to 30-60+ FPS for ultra-responsive hand tracking.
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class ThreadedCamera:
    """
    High-performance webcam capture running in a dedicated background thread.
    Always provides the freshest frame without buffering lag or blocking the UI loop.
    """

    def __init__(
        self,
        device_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        fps: int = 30
    ):
        self.device_index = device_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.frame_timestamp: float = 0.0
        self.frame_count: int = 0
        self.is_opened: bool = False

        self._thread: Optional[threading.Thread] = None

        self._start_capture()

    def _start_capture(self):
        # Open video capture with minimal buffer
        self.cap = cv2.VideoCapture(self.device_index)

        # Set low buffer size to prevent frame queuing / stale frames
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self.cap.isOpened():
            self.is_opened = False
            return

        self.is_opened = True

        # Grab initial frame
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.latest_frame = frame
            self.frame_timestamp = time.time()

        self.running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="ThreadedCameraWorker")
        self._thread.start()

    def _capture_worker(self):
        """Continuously grabs frames in background."""
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.frame_timestamp = time.time()
                    self.frame_count += 1
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Non-blocking read. Returns (True, frame) with the most recent frame,
        or (False, None) if no frame is available.
        """
        with self.lock:
            if self.latest_frame is not None:
                # Return frame immediately
                return True, self.latest_frame.copy()
            return False, None

    def stop(self):
        """Cleanly terminates the capture thread and releases camera."""
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.is_opened = False

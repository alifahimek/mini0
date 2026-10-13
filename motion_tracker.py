"""
Motion Tracker Module.
Tracks the whole-hand center across consecutive frames, applies EMA smoothing,
maintains a timestamped sliding trajectory window, and determines instantaneous motion status.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from config import MotionConfig
from hand_detector import HandLandmarkData


@dataclass
class MotionPoint:
    """A timestamped hand-center coordinate."""
    timestamp: float
    x: float            # Smoothed pixel x
    y: float            # Smoothed pixel y
    norm_x: float       # Normalized x (0.0 to 1.0)
    norm_y: float       # Normalized y (0.0 to 1.0)
    hand_span: float    # Hand scale in pixels
    is_fist_closed: bool = False
    fist_confidence: float = 0.0


class MotionTracker:
    """
    Maintains a sliding temporal window of the whole-hand center.
    Filters jitter using Exponential Moving Average (EMA) and computes
    continuous velocity and tracking status.
    """

    def __init__(self, config: MotionConfig):
        self.config = config
        self.history: Deque[MotionPoint] = deque()

        # Smoothing state
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._smooth_norm_x: Optional[float] = None
        self._smooth_norm_y: Optional[float] = None

        # Real-time motion status
        self.is_tracking: bool = False
        self.lost_frames: int = 0
        self.current_is_fist_closed: bool = False
        self.current_fist_confidence: float = 0.0
        self.current_velocity_x_norm: float = 0.0  # Normalized widths / sec
        self.current_velocity_y_norm: float = 0.0
        self.current_velocity_x_px: float = 0.0    # Pixels / sec
        self.current_velocity_y_px: float = 0.0
        self.motion_status: str = "No Hand Detected"

    def update(
        self,
        hand_data: Optional[HandLandmarkData],
        timestamp: Optional[float] = None,
        frame_width: int = 640,
        frame_height: int = 480
    ) -> str:
        """
        Updates the tracker with a new frame detection.
        Returns the current motion status string.
        """
        if timestamp is None:
            timestamp = time.time()

        if hand_data is None:
            self.lost_frames += 1
            grace = getattr(self.config, "lost_frames_grace", 2)
            if self.lost_frames <= grace and len(self.history) > 0:
                # Grace period: maintain recent motion trajectory during fast motion
                self._prune_history(timestamp)
                return self.motion_status

            # Hand lost beyond grace period
            self.is_tracking = False
            self.motion_status = "No Hand Detected"
            self.current_velocity_x_norm = 0.0
            self.current_velocity_y_norm = 0.0
            self.current_velocity_x_px = 0.0
            self.current_velocity_y_px = 0.0
            self._prune_history(timestamp)
            return self.motion_status

        # Hand is detected: reset lost frames
        self.lost_frames = 0
        self.current_is_fist_closed = hand_data.is_fist_closed
        self.current_fist_confidence = hand_data.fist_confidence
        raw_x_px, raw_y_px = hand_data.palm_center_px
        raw_x_norm, raw_y_norm = hand_data.palm_center_norm
        alpha = self.config.ema_alpha

        # Apply EMA smoothing to prevent jitter
        if self._smooth_x is None:
            self._smooth_x = raw_x_px
            self._smooth_y = raw_y_px
            self._smooth_norm_x = raw_x_norm
            self._smooth_norm_y = raw_y_norm
        else:
            self._smooth_x = alpha * raw_x_px + (1.0 - alpha) * self._smooth_x
            self._smooth_y = alpha * raw_y_px + (1.0 - alpha) * self._smooth_y
            self._smooth_norm_x = alpha * raw_x_norm + (1.0 - alpha) * self._smooth_norm_x
            self._smooth_norm_y = alpha * raw_y_norm + (1.0 - alpha) * self._smooth_norm_y

        point = MotionPoint(
            timestamp=timestamp,
            x=self._smooth_x,
            y=self._smooth_y,
            norm_x=self._smooth_norm_x,
            norm_y=self._smooth_norm_y,
            hand_span=hand_data.hand_span_px,
            is_fist_closed=hand_data.is_fist_closed,
            fist_confidence=hand_data.fist_confidence
        )
        self.history.append(point)
        self.is_tracking = True

        # Prune points outside time window
        self._prune_history(timestamp)

        # Compute instantaneous velocity over recent short baseline (last ~0.08 to 0.15s)
        self._compute_instantaneous_velocity(frame_width, frame_height)

        # Classify continuous motion state
        v_thresh = self.config.status_velocity_ratio
        if self.current_velocity_x_norm > v_thresh:
            self.motion_status = "Moving Right"
        elif self.current_velocity_x_norm < -v_thresh:
            self.motion_status = "Moving Left"
        else:
            self.motion_status = "Tracking Hand"

        return self.motion_status

    def _prune_history(self, current_time: float):
        """Remove trajectory points that fall outside the configured sliding window."""
        max_age = self.config.time_window_seconds
        while self.history and (current_time - self.history[0].timestamp) > max_age:
            self.history.popleft()

    def _compute_instantaneous_velocity(self, frame_width: int, frame_height: int):
        """Calculates smoothed short-term velocity for real-time status display."""
        if len(self.history) < 2:
            self.current_velocity_x_norm = 0.0
            self.current_velocity_y_norm = 0.0
            self.current_velocity_x_px = 0.0
            self.current_velocity_y_px = 0.0
            return

        # Take difference between newest and an older point ~0.10s ago
        newest = self.history[-1]
        baseline = self.history[0]
        # Find a point ~0.1s in the past if window is larger
        for pt in reversed(self.history):
            if (newest.timestamp - pt.timestamp) >= 0.08:
                baseline = pt
                break

        dt = newest.timestamp - baseline.timestamp
        if dt < 0.01:
            return

        dx_px = newest.x - baseline.x
        dy_px = newest.y - baseline.y
        dx_norm = newest.norm_x - baseline.norm_x
        dy_norm = newest.norm_y - baseline.norm_y

        self.current_velocity_x_px = dx_px / dt
        self.current_velocity_y_px = dy_px / dt
        self.current_velocity_x_norm = dx_norm / dt
        self.current_velocity_y_norm = dy_norm / dt

    def get_history(self) -> List[MotionPoint]:
        """Returns snapshot of current trajectory window."""
        return list(self.history)

    def get_trail_points(self, max_count: int = 16) -> List[Tuple[int, int]]:
        """Returns pixel points suitable for rendering the motion trail."""
        pts = [(int(p.x), int(p.y)) for p in self.history]
        if len(pts) > max_count:
            pts = pts[-max_count:]
        return pts

    def get_smoothed_center(self) -> Optional[Tuple[float, float]]:
        """Returns the current smoothed palm center in pixels, if tracking."""
        if self._smooth_x is not None and self.is_tracking:
            return (self._smooth_x, self._smooth_y)
        return None

    def reset(self):
        """Clears smoothing state and motion history (called on gesture trigger)."""
        self.history.clear()
        self.lost_frames = 0
        self._smooth_x = None
        self._smooth_y = None
        self._smooth_norm_x = None
        self._smooth_norm_y = None
        self.current_velocity_x_norm = 0.0
        self.current_velocity_y_norm = 0.0
        self.current_velocity_x_px = 0.0
        self.current_velocity_y_px = 0.0

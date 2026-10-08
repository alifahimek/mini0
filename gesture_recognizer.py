"""
Gesture Recognizer Module.
Detects whole-hand LEFT SWIPE and RIGHT SWIPE gestures using rule-based trajectory analysis.
Includes cooldown/debouncing, horizontal dominance check, and directional monotonicity.
No ML classifier required; fast and deterministic.
"""

import time
from enum import Enum
from typing import List, Optional, Tuple

from config import MotionConfig
from motion_tracker import MotionPoint


class Gesture(Enum):
    NONE = "None"
    RIGHT_SWIPE = "Right Swipe -> Next Slide"
    LEFT_SWIPE = "Left Swipe -> Previous Slide"


class SwipeRecognizer:
    """
    Analyzes hand-center trajectory to detect Left and Right swipes.
    Guarantees:
    - Entire hand used as input (pose-agnostic)
    - Rejection of vertical/diagonal/accidental movements
    - Debounced/cooldown protected to prevent multiple slides per swipe
    - Return-stroke immunity by clearing trajectory buffer upon trigger
    """

    def __init__(self, config: MotionConfig):
        self.config = config
        self.last_swipe_time: float = 0.0
        self.last_gesture: Gesture = Gesture.NONE
        self.last_swipe_event_time: float = 0.0

    def is_in_cooldown(self, current_time: Optional[float] = None) -> bool:
        """Returns True if debounce cooldown is currently active."""
        if current_time is None:
            current_time = time.time()
        return (current_time - self.last_swipe_time) < self.config.cooldown_seconds

    def get_remaining_cooldown(self, current_time: Optional[float] = None) -> float:
        """Returns remaining cooldown seconds (0.0 if ready)."""
        if current_time is None:
            current_time = time.time()
        elapsed = current_time - self.last_swipe_time
        remaining = self.config.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def process(self, history: List[MotionPoint], current_time: Optional[float] = None) -> Gesture:
        """
        Analyzes trajectory history and detects if a valid swipe occurred.
        Returns Gesture.RIGHT_SWIPE, Gesture.LEFT_SWIPE, or Gesture.NONE.
        """
        if current_time is None:
            current_time = time.time()

        # 1. Debounce check
        if (current_time - self.last_swipe_time) < self.config.cooldown_seconds:
            return Gesture.NONE

        # 2. Minimum trajectory requirements
        if len(history) < 4:
            return Gesture.NONE

        span_duration = history[-1].timestamp - history[0].timestamp
        if span_duration < 0.09:
            # Trajectory too brief to evaluate a deliberate gesture
            return Gesture.NONE

        # 3. Check for Right Swipe and Left Swipe candidates
        right_detected = self._check_swipe_direction(history, direction=+1)
        if right_detected:
            self.last_swipe_time = current_time
            self.last_gesture = Gesture.RIGHT_SWIPE
            self.last_swipe_event_time = current_time
            return Gesture.RIGHT_SWIPE

        left_detected = self._check_swipe_direction(history, direction=-1)
        if left_detected:
            self.last_swipe_time = current_time
            self.last_gesture = Gesture.LEFT_SWIPE
            self.last_swipe_event_time = current_time
            return Gesture.LEFT_SWIPE

        return Gesture.NONE

    def _check_swipe_direction(self, history: List[MotionPoint], direction: int) -> bool:
        """
        Evaluates whether trajectory exhibits a valid swipe.
        direction: +1 for Right Swipe, -1 for Left Swipe.
        Uses dual-strategy analysis:
        1. Real-time stroke ending at current frame (instant zero-delay trigger)
        2. Extrema pair analysis (captures completed swipes that decelerated)
        """
        if len(history) < 3:
            return False

        min_disp = self.config.min_displacement_ratio
        min_vel = self.config.min_velocity_ratio
        horiz_ratio = self.config.horizontal_dominance_ratio
        max_dt = self.config.time_window_seconds

        # Strategy 1: Real-time stroke ending at current frame
        p_now = history[-1]
        for p_start in reversed(history[:-1]):
            dt = p_now.timestamp - p_start.timestamp
            if dt > max_dt:
                break
            if dt >= 0.06:
                dx = (p_now.norm_x - p_start.norm_x) * direction
                dy = abs(p_now.norm_y - p_start.norm_y)
                if dx >= min_disp:
                    velocity_x = dx / dt
                    if velocity_x >= min_vel and dx >= (horiz_ratio * dy):
                        if self._is_monotonic(p_start, p_now, history, direction):
                            return True

        # Strategy 2: Extrema pair in sliding window
        if direction > 0:
            start_cand = min(history, key=lambda p: p.norm_x)
            end_cand = max(history, key=lambda p: p.norm_x)
        else:
            start_cand = max(history, key=lambda p: p.norm_x)
            end_cand = min(history, key=lambda p: p.norm_x)

        if end_cand.timestamp > start_cand.timestamp:
            dt = end_cand.timestamp - start_cand.timestamp
            if 0.06 <= dt <= max_dt:
                dx = (end_cand.norm_x - start_cand.norm_x) * direction
                dy = abs(end_cand.norm_y - start_cand.norm_y)
                if dx >= min_disp:
                    velocity_x = dx / dt
                    if velocity_x >= min_vel and dx >= (horiz_ratio * dy):
                        if self._is_monotonic(start_cand, end_cand, history, direction):
                            return True

        return False

    def _is_monotonic(
        self,
        p_start: MotionPoint,
        p_end: MotionPoint,
        history: List[MotionPoint],
        direction: int
    ) -> bool:
        """Verifies that the sub-trajectory is predominantly advancing in swipe direction."""
        sub_traj = [
            p for p in history
            if p_start.timestamp <= p.timestamp <= p_end.timestamp
        ]
        if len(sub_traj) < 3:
            return True

        consistent_steps = 0
        total_steps = len(sub_traj) - 1
        for i in range(total_steps):
            step_dx = (sub_traj[i + 1].norm_x - sub_traj[i].norm_x) * direction
            # Allow slight sensor noise or momentary pause
            if step_dx >= -0.020:
                consistent_steps += 1

        consistency = consistent_steps / float(total_steps)
        return consistency >= self.config.direction_consistency_threshold

    def reset_cooldown(self):
        """Manually reset the cooldown timer."""
        self.last_swipe_time = 0.0

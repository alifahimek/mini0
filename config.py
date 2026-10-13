"""
Configuration parameters for Hand-Gesture PowerPoint Presentation System.
All thresholds, camera settings, colors, and keybindings are centralized here.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    mirror_feed: bool = True  # Natural mirror mode: moving hand right moves right on screen


@dataclass
class MotionConfig:
    # Time window for trajectory analysis (seconds)
    time_window_seconds: float = 0.40

    # Minimum displacement along X-axis as a fraction of frame width
    # 0.12 on 640px width = ~77 pixels (comfortable, effortless swipe)
    min_displacement_ratio: float = 0.12

    # Minimum velocity along X-axis (frame-width fraction per second)
    # 0.28 frame widths / sec allows both swift flicks and natural-paced hand swipes
    min_velocity_ratio: float = 0.28

    # Horizontal dominance ratio: |delta_x| must be >= horizontal_ratio * |delta_y|
    # 1.15 accommodates natural arm arc curvature (especially inward left swipes)
    # while rejecting purely vertical hand waving or head nods
    horizontal_dominance_ratio: float = 1.15

    # Direction consistency ratio: fraction of frame-to-frame movements in swipe direction
    direction_consistency_threshold: float = 0.55

    # Cooldown duration after a triggered swipe (seconds) to prevent double slides
    # 0.90s is snappy and responsive for presentation control
    cooldown_seconds: float = 0.90

    # Exponential Moving Average (EMA) smoothing factor for hand center
    # 0.80 provides rapid reaction with high-frequency jitter suppression
    ema_alpha: float = 0.80

    # Velocity threshold to display real-time "Moving Right" / "Moving Left" status
    status_velocity_ratio: float = 0.12

    # Grace frames to bridge momentary tracking dropouts during fast swings
    lost_frames_grace: int = 2

    # Closed Fist Requirement:
    # Only allow slide transitions when the hand is shaped into a closed fist!
    require_closed_fist: bool = True

    # Finger curl ratio threshold for closed fist classification
    fist_curl_threshold: float = 0.58


@dataclass
class KeyConfig:
    # Keys sent via PyAutoGUI
    next_slide_key: str = "right"
    prev_slide_key: str = "left"

    # Default whether PyAutoGUI actually emits key presses
    enable_key_emission: bool = True


@dataclass
class VisualConfig:
    # Color palette (BGR for OpenCV)
    COLOR_BG_DARK: Tuple[int, int, int] = (20, 20, 24)
    COLOR_BORDER: Tuple[int, int, int] = (60, 60, 70)
    COLOR_TEXT: Tuple[int, int, int] = (240, 240, 245)
    COLOR_TEXT_DIM: Tuple[int, int, int] = (160, 160, 170)

    # Status indicator colors
    COLOR_NO_HAND: Tuple[int, int, int] = (70, 70, 180)        # Muted red
    COLOR_HAND_DETECTED: Tuple[int, int, int] = (180, 180, 60)  # Muted cyan
    COLOR_FIST_CLOSED: Tuple[int, int, int] = (0, 255, 128)     # Vibrant neon green
    COLOR_FIST_OPEN: Tuple[int, int, int] = (120, 120, 180)     # Muted amber/red for open/locked hand
    COLOR_TRACKING: Tuple[int, int, int] = (80, 200, 80)        # Vibrant green
    COLOR_MOVING_RIGHT: Tuple[int, int, int] = (255, 180, 40)   # Bright Cyan/Sky
    COLOR_MOVING_LEFT: Tuple[int, int, int] = (40, 180, 255)    # Amber/Orange
    COLOR_SWIPE_TRIGGER: Tuple[int, int, int] = (0, 230, 255)   # Gold/Yellow

    # Hand skeleton visual settings
    COLOR_BONE: Tuple[int, int, int] = (255, 200, 0)            # Cyan bones
    COLOR_JOINT: Tuple[int, int, int] = (0, 165, 255)           # Orange joints
    COLOR_PALM_CENTER: Tuple[int, int, int] = (0, 255, 128)     # Neon lime for palm center
    COLOR_TRAIL: Tuple[int, int, int] = (255, 100, 50)          # Motion trail color

    # Motion trail settings
    trail_max_points: int = 16
    swipe_banner_duration: float = 0.85  # How long the swipe trigger banner stays visible


class Config:
    camera = CameraConfig()
    motion = MotionConfig()
    key = KeyConfig()
    visual = VisualConfig()

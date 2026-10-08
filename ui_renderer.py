"""
UI Renderer Module.
Draws hand skeleton, landmarks, whole-hand palm center marker, motion trail,
heads-up display (HUD), cooldown indicator, and swipe feedback banners on the OpenCV frame.
"""

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import Config, VisualConfig
from hand_detector import HAND_CONNECTIONS, HandLandmarkData
from gesture_recognizer import Gesture, SwipeRecognizer
from motion_tracker import MotionTracker
from presentation_controller import PresentationController


class UIRenderer:
    """
    Renders visual overlays on top of the OpenCV camera feed:
    - 21-point hand skeleton with distinct joints and bones
    - Whole-hand palm center highlight (independent of finger configuration)
    - Real-time motion trail
    - Top HUD bar with detection/tracking status and PowerPoint state
    - Cooldown countdown bar
    - Swipe triggered notification banners
    - Bottom control hints
    """

    def __init__(self, config: Config):
        self.config = config
        self.visual = config.visual

        # Banner state
        self.banner_text: str = ""
        self.banner_color: Tuple[int, int, int] = (0, 255, 0)
        self.banner_until: float = 0.0
        self.banner_subtext: str = ""

    def show_swipe_banner(self, gesture: Gesture):
        """Triggers a visual banner for a detected swipe event."""
        now = time.time()
        self.banner_until = now + self.visual.swipe_banner_duration

        if gesture == Gesture.RIGHT_SWIPE:
            self.banner_text = "RIGHT SWIPE -> NEXT SLIDE"
            self.banner_subtext = ">> Key [RIGHT] Dispatched >>"
            self.banner_color = (0, 230, 255)  # Bright Gold / Yellow
        elif gesture == Gesture.LEFT_SWIPE:
            self.banner_text = "LEFT SWIPE -> PREVIOUS SLIDE"
            self.banner_subtext = "<< Key [LEFT] Dispatched <<"
            self.banner_color = (255, 140, 0)  # Bright Orange / Blue in BGR

    def render(
        self,
        frame: np.ndarray,
        hand_data: Optional[HandLandmarkData],
        motion_tracker: MotionTracker,
        swipe_recognizer: SwipeRecognizer,
        presentation_controller: PresentationController,
        fps: float = 0.0,
        show_landmarks: bool = True
    ) -> np.ndarray:
        """Draws all visual layers onto the provided frame."""
        h, w = frame.shape[:2]
        output = frame.copy()

        # 1. Draw Hand Skeleton & Landmarks
        if show_landmarks and hand_data is not None:
            self._draw_hand_skeleton(output, hand_data)
            self._draw_palm_center(output, hand_data)

        # 2. Draw Motion Trail
        self._draw_motion_trail(output, motion_tracker)

        # 3. Draw Top HUD Bar
        self._draw_hud_header(
            output,
            hand_data,
            motion_tracker,
            swipe_recognizer,
            presentation_controller,
            fps,
            w
        )

        # 4. Draw Swipe Trigger Notification Banner if active
        now = time.time()
        if now < self.banner_until:
            self._draw_swipe_banner(output, w, h)

        # 5. Draw Bottom Controls Guide
        self._draw_bottom_footer(output, w, h, presentation_controller)

        return output

    def _draw_hand_skeleton(self, frame: np.ndarray, hand_data: HandLandmarkData):
        """Draws 21 landmarks and bone connections."""
        pts = hand_data.landmarks_px

        # Draw bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = pts[start_idx]
            p2 = pts[end_idx]
            cv2.line(frame, p1, p2, self.visual.COLOR_BONE, 2, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(pts):
            radius = 3 if i not in (0, 4, 8, 12, 16, 20) else 5
            color = self.visual.COLOR_JOINT
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw subtle bounding box
        bx1, by1, bx2, by2 = hand_data.bbox_px
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 100), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{hand_data.handedness} Hand",
            (bx1, max(15, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 200),
            1,
            cv2.LINE_AA
        )

    def _draw_palm_center(self, frame: np.ndarray, hand_data: HandLandmarkData):
        """Draws prominent whole-hand palm center marker."""
        cx = int(round(hand_data.palm_center_px[0]))
        cy = int(round(hand_data.palm_center_px[1]))
        color = self.visual.COLOR_PALM_CENTER

        # Outer ring
        cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)
        # Inner dot
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        # Crosshair
        cv2.line(frame, (cx - 16, cy), (cx + 16, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 16), (cx, cy + 16), color, 1, cv2.LINE_AA)

        # Label
        cv2.putText(
            frame,
            "Hand Center",
            (cx + 14, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    def _draw_motion_trail(self, frame: np.ndarray, motion_tracker: MotionTracker):
        """Draws fading trajectory trail of hand center."""
        trail = motion_tracker.get_trail_points(self.visual.trail_max_points)
        n = len(trail)
        if n < 2:
            return

        for i in range(1, n):
            p1 = trail[i - 1]
            p2 = trail[i]
            # Increasing opacity & thickness towards newest point
            progress = i / float(n)
            thickness = max(1, int(progress * 4))
            alpha_color = tuple(int(c * progress) for c in self.visual.COLOR_TRAIL)
            cv2.line(frame, p1, p2, alpha_color, thickness, cv2.LINE_AA)
            if i == n - 1:
                cv2.circle(frame, p2, 5, self.visual.COLOR_TRAIL, -1, cv2.LINE_AA)

    def _draw_hud_header(
        self,
        frame: np.ndarray,
        hand_data: Optional[HandLandmarkData],
        motion_tracker: MotionTracker,
        swipe_recognizer: SwipeRecognizer,
        presentation_controller: PresentationController,
        fps: float,
        w: int
    ):
        """Draws top status bar with dark background."""
        header_height = 58
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), self.visual.COLOR_BG_DARK, -1)
        cv2.line(overlay, (0, header_height), (w, header_height), self.visual.COLOR_BORDER, 1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # 1. Determine Tracking Status & Badge Color
        if hand_data is None:
            status_text = "No Hand Detected"
            status_color = self.visual.COLOR_NO_HAND
        else:
            status_text = motion_tracker.motion_status
            if status_text == "Moving Right":
                status_color = self.visual.COLOR_MOVING_RIGHT
            elif status_text == "Moving Left":
                status_color = self.visual.COLOR_MOVING_LEFT
            else:
                status_text = "Tracking Hand"
                status_color = self.visual.COLOR_TRACKING

        # Status badge
        cv2.circle(frame, (18, 22), 7, status_color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            status_text.upper(),
            (32, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            status_color,
            2,
            cv2.LINE_AA
        )

        # Hand architecture note
        cv2.putText(
            frame,
            "Mode: Whole-Hand / Pose-Agnostic",
            (32, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.visual.COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA
        )

        # 2. Middle: Cooldown Status / Progress Bar
        cooldown_left = swipe_recognizer.get_remaining_cooldown()
        bar_x = w // 2 - 80
        bar_y = 16
        bar_w = 160
        bar_h = 10

        if cooldown_left > 0.0:
            total_cd = self.config.motion.cooldown_seconds
            fraction = 1.0 - (cooldown_left / total_cd)
            fill_w = int(bar_w * fraction)

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 180, 255), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 110), 1)

            cv2.putText(
                frame,
                f"Cooldown: {cooldown_left:.1f}s",
                (bar_x, bar_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 180, 255),
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame,
                "Ready for Swipe",
                (bar_x + 20, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 230, 100),
                1,
                cv2.LINE_AA
            )

        # 3. Right: PowerPoint emission state & FPS
        ppt_status = "PPT: ACTIVE" if presentation_controller.enabled else "PPT: MUTED"
        ppt_color = (100, 255, 100) if presentation_controller.enabled else (120, 120, 220)
        cv2.putText(
            frame,
            ppt_status,
            (w - 170, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ppt_color,
            1,
            cv2.LINE_AA
        )

        stats_str = f"Next:{presentation_controller.next_slide_count}  Prev:{presentation_controller.prev_slide_count}  FPS:{fps:.0f}"
        cv2.putText(
            frame,
            stats_str,
            (w - 200, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.visual.COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA
        )

    def _draw_swipe_banner(self, frame: np.ndarray, w: int, h: int):
        """Draws animated alert banner when a swipe is triggered."""
        box_w = 460
        box_h = 75
        bx1 = (w - box_w) // 2
        by1 = 80
        bx2 = bx1 + box_w
        by2 = by1 + box_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 20), -1)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), self.banner_color, 2)
        cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

        # Title
        cv2.putText(
            frame,
            self.banner_text,
            (bx1 + 22, by1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            self.banner_color,
            2,
            cv2.LINE_AA
        )

        # Subtext
        cv2.putText(
            frame,
            self.banner_subtext,
            (bx1 + 80, by1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _draw_bottom_footer(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        presentation_controller: PresentationController
    ):
        """Draws keybindings help at the bottom."""
        footer_height = 26
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - footer_height), (w, h), (15, 15, 18), -1)
        cv2.line(overlay, (0, h - footer_height), (w, h - footer_height), (40, 40, 45), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        hint_text = "[Q] Quit  |  [C] Toggle PPT Control  |  [M] Toggle Mirror  |  [R] Reset Trajectory  |  [H] Toggle Skeleton"
        cv2.putText(
            frame,
            hint_text,
            (14, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (180, 180, 190),
            1,
            cv2.LINE_AA
        )

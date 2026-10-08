"""
Hand Detector Module.
Provides robust whole-hand detection using MediaPipe across different hand
orientations, rotations, palm-facing, back-facing, and tilted hand poses.

Compatible with both MediaPipe Tasks API (v1.0+) and Legacy Solutions API (v0.10.x).
"""

import os
import sys
import math
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Official MediaPipe Hand Landmarker model bundle URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

# Standard MediaPipe 21 Hand Landmark Connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles (Metacarpophalangeal arch)
    (5, 9), (9, 13), (13, 17)
]

# Rigid palm landmarks: Wrist (0), Thumb CMC (1), Index MCP (5),
# Middle MCP (9), Ring MCP (13), Pinky MCP (17).
# These points define the stable palm body regardless of finger curls or rotations.
PALM_LANDMARK_INDICES = [0, 1, 5, 9, 13, 17]


@dataclass
class HandLandmarkData:
    """Structured whole-hand tracking data for a detected hand."""
    # 21 landmarks in normalized coordinates (0.0 to 1.0)
    landmarks_norm: List[Tuple[float, float, float]]

    # 21 landmarks in pixel coordinates (x, y)
    landmarks_px: List[Tuple[int, int]]

    # Handedness label ('Left', 'Right', or 'Hand')
    handedness: str

    # Stable whole-hand palm center in pixel coordinates (x, y)
    palm_center_px: Tuple[float, float]

    # Stable whole-hand palm center in normalized coordinates (x, y)
    palm_center_norm: Tuple[float, float]

    # Centroid of all 21 landmarks in pixels
    all_landmarks_centroid_px: Tuple[float, float]

    # Bounding box of hand: (min_x, min_y, max_x, max_y) in pixels
    bbox_px: Tuple[int, int, int, int]

    # Hand scale: distance between wrist (0) and middle MCP (9) in pixels
    hand_span_px: float

    # Raw wrist pixel coordinates
    wrist_px: Tuple[int, int]


class HandDetector:
    """
    Robust whole-hand detector supporting both MediaPipe Tasks API and Legacy Solutions API.
    Extracts complete 21 landmarks and computes a rotation/pose-invariant palm center.
    """

    def __init__(self, model_path: Optional[str] = None, max_num_hands: int = 1, min_detection_confidence: float = 0.35):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.engine_type: str = "unknown"
        self._landmarker = None
        self._mp_hands_solution = None

        self._initialize_detector(model_path)

    def _ensure_model_file(self, model_path: Optional[str]) -> str:
        """Ensure the .task model file is downloaded and present locally."""
        if model_path and os.path.exists(model_path):
            return model_path

        # Check default location in current directory or module directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(base_dir, MODEL_FILENAME)

        if not os.path.exists(target_path):
            print(f"[HandDetector] Model file not found at {target_path}. Downloading from Google...")
            try:
                urllib.request.urlretrieve(MODEL_URL, target_path)
                print(f"[HandDetector] Successfully downloaded model to {target_path} ({os.path.getsize(target_path)} bytes)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download MediaPipe hand model from {MODEL_URL}: {e}"
                )
        return target_path

    def _initialize_detector(self, model_path: Optional[str]):
        """Initialize MediaPipe detector using Tasks API or Solutions API."""
        import mediapipe as mp

        # Check if legacy mp.solutions is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self._mp_hands_solution = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence
                )
                self.engine_type = "solutions_api"
                print("[HandDetector] Initialized using MediaPipe Solutions API (v0.10.x)")
                return
            except Exception as e:
                print(f"[HandDetector] Legacy solutions API failed: {e}. Falling back to Tasks API.")

        # Fallback / Modern: MediaPipe Tasks API (MediaPipe v1.0+)
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            resolved_model_path = self._ensure_model_file(model_path)
            base_options = python.BaseOptions(model_asset_path=resolved_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_detection_confidence
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self.engine_type = "tasks_api"
            print("[HandDetector] Initialized using MediaPipe Tasks API (v1.0+)")
        except Exception as e:
            raise RuntimeError(f"[HandDetector] Failed to initialize MediaPipe HandLandmarker: {e}")

    def detect(self, bgr_frame: np.ndarray) -> Optional[HandLandmarkData]:
        """
        Detect hand in a BGR frame.
        Returns HandLandmarkData for the primary detected hand, or None if no hand is detected.
        """
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        if self.engine_type == "tasks_api":
            return self._detect_tasks_api(rgb_frame, w, h)
        elif self.engine_type == "solutions_api":
            return self._detect_solutions_api(rgb_frame, w, h)
        return None

    def _detect_tasks_api(self, rgb_frame: np.ndarray, w: int, h: int) -> Optional[HandLandmarkData]:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        # Primary hand
        first_hand = result.hand_landmarks[0]
        handedness_label = "Hand"
        if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
            handedness_label = result.handedness[0][0].category_name

        landmarks_norm: List[Tuple[float, float, float]] = []
        landmarks_px: List[Tuple[int, int]] = []

        for lm in first_hand:
            landmarks_norm.append((lm.x, lm.y, lm.z))
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            landmarks_px.append((px, py))

        return self._build_hand_data(landmarks_norm, landmarks_px, handedness_label, w, h)

    def _detect_solutions_api(self, rgb_frame: np.ndarray, w: int, h: int) -> Optional[HandLandmarkData]:
        results = self._mp_hands_solution.process(rgb_frame)

        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) == 0:
            return None

        first_hand = results.multi_hand_landmarks[0]
        handedness_label = "Hand"
        if results.multi_handedness and len(results.multi_handedness) > 0:
            handedness_label = results.multi_handedness[0].classification[0].label

        landmarks_norm: List[Tuple[float, float, float]] = []
        landmarks_px: List[Tuple[int, int]] = []

        for lm in first_hand.landmark:
            landmarks_norm.append((lm.x, lm.y, lm.z))
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            landmarks_px.append((px, py))

        return self._build_hand_data(landmarks_norm, landmarks_px, handedness_label, w, h)

    def _build_hand_data(
        self,
        landmarks_norm: List[Tuple[float, float, float]],
        landmarks_px: List[Tuple[int, int]],
        handedness: str,
        w: int,
        h: int
    ) -> HandLandmarkData:
        """
        Computes robust whole-hand metrics:
        - Palm center: Mean of wrist (0) and 5 MCP knuckles (1, 5, 9, 13, 17)
        - All landmarks centroid
        - Bounding box
        - Hand span (scale)
        """
        # 1. Palm Center Calculation (Rigid Anatomical Core)
        # Using Wrist (0), Thumb CMC (1), Index MCP (5), Middle MCP (9), Ring MCP (13), Pinky MCP (17)
        palm_px_points = [landmarks_px[i] for i in PALM_LANDMARK_INDICES]
        palm_center_x = float(sum(p[0] for p in palm_px_points) / len(palm_px_points))
        palm_center_y = float(sum(p[1] for p in palm_px_points) / len(palm_px_points))
        palm_center_px = (palm_center_x, palm_center_y)

        palm_norm_points = [landmarks_norm[i] for i in PALM_LANDMARK_INDICES]
        palm_norm_x = float(sum(p[0] for p in palm_norm_points) / len(palm_norm_points))
        palm_norm_y = float(sum(p[1] for p in palm_norm_points) / len(palm_norm_points))
        palm_center_norm = (palm_norm_x, palm_norm_y)

        # 2. Overall centroid across all 21 points
        all_x = float(sum(p[0] for p in landmarks_px) / len(landmarks_px))
        all_y = float(sum(p[1] for p in landmarks_px) / len(landmarks_px))
        all_centroid_px = (all_x, all_y)

        # 3. Bounding Box
        xs = [p[0] for p in landmarks_px]
        ys = [p[1] for p in landmarks_px]
        bbox_px = (min(xs), min(ys), max(xs), max(ys))

        # 4. Hand Span / Scale (distance between Wrist (0) and Middle MCP (9))
        w0 = landmarks_px[0]
        m9 = landmarks_px[9]
        hand_span = math.hypot(m9[0] - w0[0], m9[1] - w0[1])
        if hand_span < 1.0:
            hand_span = 1.0

        return HandLandmarkData(
            landmarks_norm=landmarks_norm,
            landmarks_px=landmarks_px,
            handedness=handedness,
            palm_center_px=palm_center_px,
            palm_center_norm=palm_center_norm,
            all_landmarks_centroid_px=all_centroid_px,
            bbox_px=bbox_px,
            hand_span_px=hand_span,
            wrist_px=landmarks_px[0]
        )

    def close(self):
        """Release MediaPipe resources cleanly."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

        if self._mp_hands_solution is not None:
            try:
                self._mp_hands_solution.close()
            except Exception:
                pass
            self._mp_hands_solution = None

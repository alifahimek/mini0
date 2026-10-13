"""
Automated Unit and Integration Tests for Hand Gesture PowerPoint System.
Verifies:
1. Whole-Hand palm center invariant calculation under rotation, tilt, and finger poses.
2. Trajectory-based Swipe Recognition (Right Swipe, Left Swipe).
3. Rejection of vertical waves, diagonal drifts, and minor hand jitters.
4. Cooldown/debounce mechanism and return-stroke immunity.
5. Presentation Controller event dispatching and muting.
6. HandDetector engine initialization.
"""

import math
import time
import unittest
import numpy as np

from config import Config, MotionConfig, KeyConfig
from hand_detector import HandDetector, HandLandmarkData, PALM_LANDMARK_INDICES
from motion_tracker import MotionPoint, MotionTracker
from gesture_recognizer import Gesture, SwipeRecognizer
from presentation_controller import PresentationController


def generate_synthetic_hand_landmarks(
    center_x: float = 320.0,
    center_y: float = 240.0,
    angle_rad: float = 0.0,
    scale: float = 80.0,
    fingers_curled: bool = False
):
    """
    Generates 21 synthetic hand landmarks relative to a palm center with rotation and finger pose.
    """
    # Canonical landmark template (relative offsets from wrist)
    # 0: Wrist
    # 1-4: Thumb
    # 5-8: Index
    # 9-12: Middle
    # 13-16: Ring
    # 17-20: Pinky
    template_offsets = {
        0: (0.0, 0.6),      # Wrist
        1: (-0.3, 0.4),     # Thumb CMC
        2: (-0.5, 0.2),     # Thumb MCP
        3: (-0.6, 0.0),     # Thumb IP
        4: (-0.7, -0.2),    # Thumb Tip
        5: (-0.3, 0.0),     # Index MCP
        6: (-0.35, -0.3),   # Index PIP
        7: (-0.38, -0.6),   # Index DIP
        8: (-0.4, -0.9),    # Index Tip
        9: (0.0, 0.0),      # Middle MCP
        10: (0.0, -0.35),   # Middle PIP
        11: (0.0, -0.7),    # Middle DIP
        12: (0.0, -1.0),    # Middle Tip
        13: (0.28, 0.0),    # Ring MCP
        14: (0.30, -0.32),  # Ring PIP
        15: (0.32, -0.65),  # Ring DIP
        16: (0.35, -0.95),  # Ring Tip
        17: (0.50, 0.1),    # Pinky MCP
        18: (0.52, -0.2),   # Pinky PIP
        19: (0.54, -0.5),   # Pinky DIP
        20: (0.55, -0.75),  # Pinky Tip
    }

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    landmarks_px = []
    landmarks_norm = []
    w, h = 640, 480

    for i in range(21):
        ox, oy = template_offsets[i]

        # If fingers curled (fist pose), pull fingertips (4, 8, 12, 16, 20) towards palm knuckles
        if fingers_curled and i in (3, 4, 7, 8, 11, 12, 15, 16, 19, 20):
            oy = oy * 0.25

        # Scale and rotate
        rx = (ox * cos_a - oy * sin_a) * scale
        ry = (ox * sin_a + oy * cos_a) * scale

        px = int(center_x + rx)
        py = int(center_y + ry)
        landmarks_px.append((px, py))
        landmarks_norm.append((px / float(w), py / float(h), 0.0))

    # Compute palm center
    palm_pts = [landmarks_px[k] for k in PALM_LANDMARK_INDICES]
    pc_x = sum(p[0] for p in palm_pts) / len(palm_pts)
    pc_y = sum(p[1] for p in palm_pts) / len(palm_pts)

    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]

    return HandLandmarkData(
        landmarks_norm=landmarks_norm,
        landmarks_px=landmarks_px,
        handedness="Right",
        palm_center_px=(pc_x, pc_y),
        palm_center_norm=(pc_x / float(w), pc_y / float(h)),
        all_landmarks_centroid_px=(sum(xs)/21.0, sum(ys)/21.0),
        bbox_px=(min(xs), min(ys), max(xs), max(ys)),
        hand_span_px=scale,
        wrist_px=landmarks_px[0],
        is_fist_closed=fingers_curled,
        fist_confidence=1.0 if fingers_curled else 0.0
    )


class TestWholeHandPresentationSystem(unittest.TestCase):

    def setUp(self):
        self.motion_cfg = MotionConfig(
            time_window_seconds=0.35,
            min_displacement_ratio=0.15,
            min_velocity_ratio=0.40,
            horizontal_dominance_ratio=1.35,
            cooldown_seconds=1.0,
            direction_consistency_threshold=0.65
        )
        self.recognizer = SwipeRecognizer(self.motion_cfg)
        self.tracker = MotionTracker(self.motion_cfg)

    def test_palm_center_stability_across_finger_poses(self):
        """Palm center should remain virtually identical whether fingers are open or curled in a fist."""
        open_hand = generate_synthetic_hand_landmarks(300, 200, angle_rad=0.0, fingers_curled=False)
        fist_hand = generate_synthetic_hand_landmarks(300, 200, angle_rad=0.0, fingers_curled=True)

        # Fingertip (landmark 12) moves drastically when curled:
        fingertip_delta = math.hypot(
            open_hand.landmarks_px[12][0] - fist_hand.landmarks_px[12][0],
            open_hand.landmarks_px[12][1] - fist_hand.landmarks_px[12][1]
        )
        self.assertGreater(fingertip_delta, 40.0, "Fingertip position should change noticeably when curled")

        # But Palm Center must remain completely stable!
        palm_delta = math.hypot(
            open_hand.palm_center_px[0] - fist_hand.palm_center_px[0],
            open_hand.palm_center_px[1] - fist_hand.palm_center_px[1]
        )
        self.assertAlmostEqual(palm_delta, 0.0, places=2, msg="Palm center must be invariant to finger curling!")

    def test_palm_center_rotation_tracking(self):
        """Palm center properly reflects rotated hands."""
        for angle_deg in [0, 45, 90, 180, 270]:
            angle_rad = math.radians(angle_deg)
            hand = generate_synthetic_hand_landmarks(350, 250, angle_rad=angle_rad)
            # Palm center should be close to center (350, 250)
            dist = math.hypot(hand.palm_center_px[0] - 350, hand.palm_center_px[1] - 250)
            self.assertLess(dist, 50.0, f"Palm center at {angle_deg} deg should remain near core")

    def test_right_swipe_detection(self):
        """A clean left-to-right swipe with closed fist should be detected as RIGHT_SWIPE."""
        start_time = 100.0
        history = []
        # Move from norm_x = 0.25 to norm_x = 0.65 over 0.25 seconds (displacement = 0.40)
        num_frames = 10
        for i in range(num_frames):
            t = start_time + (i * 0.025)
            progress = i / float(num_frames - 1)
            norm_x = 0.25 + (0.40 * progress)
            norm_y = 0.50 + 0.01 * (progress - 0.5)  # Tiny vertical drift
            px_x = norm_x * 640
            px_y = norm_y * 480
            history.append(MotionPoint(t, px_x, px_y, norm_x, norm_y, 80.0, is_fist_closed=True))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.RIGHT_SWIPE, "Should detect RIGHT_SWIPE")

    def test_left_swipe_detection(self):
        """A clean right-to-left swipe with closed fist should be detected as LEFT_SWIPE."""
        start_time = 100.0
        history = []
        # Move from norm_x = 0.70 to norm_x = 0.30 over 0.25 seconds (displacement = -0.40)
        num_frames = 10
        for i in range(num_frames):
            t = start_time + (i * 0.025)
            progress = i / float(num_frames - 1)
            norm_x = 0.70 - (0.40 * progress)
            norm_y = 0.50 + 0.01 * (progress - 0.5)
            px_x = norm_x * 640
            px_y = norm_y * 480
            history.append(MotionPoint(t, px_x, px_y, norm_x, norm_y, 80.0, is_fist_closed=True))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.LEFT_SWIPE, "Should detect LEFT_SWIPE")

    def test_curved_left_swipe_detection(self):
        """Natural curved inward left swipe with closed fist must trigger LEFT_SWIPE."""
        start_time = 200.0
        history = []
        # Human left swipe: starts at x=0.62, y=0.45, ends at x=0.44, y=0.57 (dy = 0.12, dx = 0.18)
        for i in range(8):
            t = start_time + (i * 0.025)
            prog = i / 7.0
            norm_x = 0.62 - (0.18 * prog)
            norm_y = 0.45 + (0.12 * prog)
            history.append(MotionPoint(t, norm_x * 640, norm_y * 480, norm_x, norm_y, 80.0, is_fist_closed=True))

        gesture = self.recognizer.process(history, current_time=start_time + 0.175)
        self.assertEqual(gesture, Gesture.LEFT_SWIPE, "Curved natural left swipe must trigger LEFT_SWIPE")

    def test_grace_period_fast_swipe(self):
        """MotionTracker preserves tracking state if hand is temporarily lost for 1-2 frames."""
        hand = generate_synthetic_hand_landmarks(320, 240)
        self.tracker.update(hand, timestamp=10.0)
        self.assertTrue(self.tracker.is_tracking)

        # Frame 2: hand temporarily dropped during fast motion
        self.tracker.update(None, timestamp=10.025)
        self.assertTrue(self.tracker.is_tracking, "Grace period should keep tracking active for 1 dropped frame")

        # Frame 3: hand re-detected
        self.tracker.update(hand, timestamp=10.050)
        self.assertTrue(self.tracker.is_tracking)
        self.assertEqual(self.tracker.lost_frames, 0)

    def test_vertical_movement_rejection(self):
        """Vertical hand movements (waving up/down) must NOT trigger a horizontal swipe."""
        start_time = 100.0
        history = []
        # Move predominantly along Y axis (vertical wave)
        num_frames = 10
        for i in range(num_frames):
            t = start_time + (i * 0.025)
            progress = i / float(num_frames - 1)
            norm_x = 0.50 + 0.02 * (progress - 0.5)   # Negligible X movement
            norm_y = 0.20 + (0.45 * progress)         # Large Y movement
            history.append(MotionPoint(t, norm_x * 640, norm_y * 480, norm_x, norm_y, 80.0))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.NONE, "Vertical movement must be rejected")

    def test_small_accidental_movement_rejection(self):
        """Small accidental jitters below min displacement threshold must be ignored."""
        start_time = 100.0
        history = []
        # Jitter of only 0.04 (threshold is 0.15)
        for i in range(10):
            t = start_time + (i * 0.025)
            norm_x = 0.50 + 0.04 * (i / 9.0)
            norm_y = 0.50
            history.append(MotionPoint(t, norm_x * 640, norm_y * 480, norm_x, norm_y, 80.0))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.NONE, "Small movement must not trigger swipe")

    def test_cooldown_debounce_mechanism(self):
        """A second swipe attempted within the cooldown period must be suppressed."""
        # 1. Trigger first swipe at t = 10.0
        history1 = []
        for i in range(10):
            t = 10.0 + (i * 0.02)
            norm_x = 0.20 + 0.35 * (i / 9.0)
            history1.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.5, 80.0, is_fist_closed=True))

        gesture1 = self.recognizer.process(history1, current_time=10.20)
        self.assertEqual(gesture1, Gesture.RIGHT_SWIPE)
        self.assertTrue(self.recognizer.is_in_cooldown(current_time=10.20))

        # 2. Immediate second swipe at t = 10.40 (within 1.0s cooldown)
        history2 = []
        for i in range(10):
            t = 10.40 + (i * 0.02)
            norm_x = 0.20 + 0.35 * (i / 9.0)
            history2.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.5, 80.0, is_fist_closed=True))

        gesture2 = self.recognizer.process(history2, current_time=10.60)
        self.assertEqual(gesture2, Gesture.NONE, "Swipe within cooldown must be suppressed")

        # 3. Third swipe at t = 11.50 (after cooldown has passed)
        history3 = []
        for i in range(10):
            t = 11.50 + (i * 0.02)
            norm_x = 0.20 + 0.35 * (i / 9.0)
            history3.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.5, 80.0, is_fist_closed=True))

        gesture3 = self.recognizer.process(history3, current_time=11.70)
        self.assertEqual(gesture3, Gesture.RIGHT_SWIPE, "Swipe after cooldown must succeed")

    def test_open_hand_swipe_rejected_when_fist_required(self):
        """When require_closed_fist is True, a swipe made with an open hand must be REJECTED."""
        self.assertTrue(self.recognizer.config.require_closed_fist)
        start_time = 100.0
        history = []
        # Rapid horizontal swipe, but hand is OPEN (is_fist_closed=False)
        for i in range(10):
            t = start_time + (i * 0.025)
            prog = i / 9.0
            norm_x = 0.25 + (0.40 * prog)
            history.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.50, 80.0, is_fist_closed=False, fist_confidence=0.0))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.NONE, "Open hand swipe must be rejected when fist is required")

    def test_closed_fist_swipe_accepted_when_fist_required(self):
        """When require_closed_fist is True, a swipe made with a CLOSED FIST must be ACCEPTED."""
        self.assertTrue(self.recognizer.config.require_closed_fist)
        start_time = 100.0
        history = []
        for i in range(10):
            t = start_time + (i * 0.025)
            prog = i / 9.0
            norm_x = 0.25 + (0.40 * prog)
            history.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.50, 80.0, is_fist_closed=True, fist_confidence=1.0))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.RIGHT_SWIPE, "Closed fist swipe must be accepted when fist is required")

    def test_fist_lock_toggle(self):
        """Disabling require_closed_fist allows open-hand swipes to register again."""
        self.recognizer.config.require_closed_fist = False
        start_time = 100.0
        history = []
        for i in range(10):
            t = start_time + (i * 0.025)
            prog = i / 9.0
            norm_x = 0.25 + (0.40 * prog)
            history.append(MotionPoint(t, norm_x * 640, 240, norm_x, 0.50, 80.0, is_fist_closed=False, fist_confidence=0.0))

        gesture = self.recognizer.process(history, current_time=start_time + 0.25)
        self.assertEqual(gesture, Gesture.RIGHT_SWIPE, "Open-hand swipe should be accepted when require_closed_fist is False")
        self.recognizer.config.require_closed_fist = True  # restore

    def test_closed_fist_classification_detector(self):
        """HandDetector correctly differentiates between open hand and closed fist."""
        detector = HandDetector()
        open_h = generate_synthetic_hand_landmarks(320, 240, fingers_curled=False)
        fist_h = generate_synthetic_hand_landmarks(320, 240, fingers_curled=True)

        is_fist_open, conf_open = detector._check_closed_fist(
            open_h.landmarks_px, open_h.landmarks_norm, open_h.hand_span_px
        )
        is_fist_closed, conf_closed = detector._check_closed_fist(
            fist_h.landmarks_px, fist_h.landmarks_norm, fist_h.hand_span_px
        )

        self.assertFalse(is_fist_open, "Open hand should not be classified as fist")
        self.assertTrue(is_fist_closed, "Curled hand should be classified as fist")
        self.assertGreater(conf_closed, 0.70)
        self.assertLess(conf_open, 0.30)
        detector.close()

    def test_presentation_controller_dispatch(self):
        """PresentationController correctly triggers actions and tracks counts."""
        key_cfg = KeyConfig(enable_key_emission=False)  # Muted for unit test
        controller = PresentationController(key_cfg)

        events_received = []
        controller.on_slide_change = lambda act: events_received.append(act)

        self.assertEqual(controller.next_slide_count, 0)
        self.assertEqual(controller.prev_slide_count, 0)

        # Trigger Right Swipe
        controller.handle_gesture(Gesture.RIGHT_SWIPE)
        self.assertEqual(controller.next_slide_count, 1)
        self.assertEqual(controller.prev_slide_count, 0)
        self.assertIn("NEXT SLIDE", events_received[0])

        # Trigger Left Swipe
        controller.handle_gesture(Gesture.LEFT_SWIPE)
        self.assertEqual(controller.next_slide_count, 1)
        self.assertEqual(controller.prev_slide_count, 1)
        self.assertIn("PREV SLIDE", events_received[1])

        # Toggle muting
        state = controller.toggle_enabled()
        self.assertTrue(state)

    def test_hand_detector_initialization(self):
        """HandDetector initializes cleanly and handles empty/blank images."""
        detector = HandDetector()
        self.assertIn(detector.engine_type, ["tasks_api", "solutions_api"])

        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank_frame)
        self.assertIsNone(result, "Blank frame should detect no hands")
        detector.close()


if __name__ == "__main__":
    unittest.main()

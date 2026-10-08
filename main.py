"""
Main Application Runner.
Hand-Gesture-Controlled PowerPoint Presentation System.

Uses whole-hand tracking with MediaPipe Hands, rule-based motion analysis,
and PyAutoGUI to navigate presentation slides via natural hand swipes.
"""

import argparse
import sys
import time

import cv2

from config import Config
from camera_stream import ThreadedCamera
from gesture_recognizer import Gesture, SwipeRecognizer
from hand_detector import HandDetector
from motion_tracker import MotionTracker
from presentation_controller import PresentationController
from ui_renderer import UIRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whole-Hand Gesture Presentation Controller (OpenCV + MediaPipe + PyAutoGUI)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable camera mirroring"
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Disable PyAutoGUI key emission (visual test mode only)"
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.20,
        help="Swipe cooldown/debounce duration in seconds (default: 1.2)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to hand_landmarker.task model file"
    )
    return parser.parse_args()


def run_app():
    args = parse_args()

    # Load configuration
    cfg = Config()
    cfg.camera.device_index = args.camera
    cfg.camera.frame_width = args.width
    cfg.camera.frame_height = args.height
    cfg.camera.mirror_feed = not args.no_mirror
    cfg.motion.cooldown_seconds = args.cooldown
    if args.mute:
        cfg.key.enable_key_emission = False

    print("=" * 65)
    print("   HAND-GESTURE POWERPOINT PRESENTATION SYSTEM")
    print("   Whole-Hand / Pose-Agnostic Gesture Controller")
    print("=" * 65)
    print(f"[Config] Camera Index   : {cfg.camera.device_index}")
    print(f"[Config] Resolution     : {cfg.camera.frame_width}x{cfg.camera.frame_height}")
    print(f"[Config] Mirror Feed    : {cfg.camera.mirror_feed}")
    print(f"[Config] Swipe Cooldown : {cfg.motion.cooldown_seconds}s")
    print(f"[Config] Key Control    : {'ENABLED (PyAutoGUI active)' if cfg.key.enable_key_emission else 'MUTED (Testing)'}")
    print("-" * 65)
    print("Interactive Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [C]          : Toggle PowerPoint keypresses (Active / Muted)")
    print("  [M]          : Toggle camera mirroring")
    print("  [R]          : Reset motion tracking buffer")
    print("  [H]          : Toggle hand landmarks skeleton visibility")
    print("=" * 65)

    # Initialize components
    print("\n[Init] Initializing MediaPipe Hand Detector...")
    try:
        detector = HandDetector(model_path=args.model_path)
    except Exception as e:
        print(f"[FATAL] Failed to initialize HandDetector: {e}")
        sys.exit(1)

    motion_tracker = MotionTracker(cfg.motion)
    swipe_recognizer = SwipeRecognizer(cfg.motion)
    presentation_controller = PresentationController(cfg.key)
    ui_renderer = UIRenderer(cfg)

    # Initialize Threaded Camera
    print(f"[Init] Opening threaded webcam capture (device {cfg.camera.device_index})...")
    camera = ThreadedCamera(
        device_index=cfg.camera.device_index,
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        fps=cfg.camera.fps
    )

    if not camera.is_opened:
        print(f"\n[ERROR] Unable to open camera device {cfg.camera.device_index}.")
        print("Please verify:")
        print(" 1. Your webcam is connected and not currently used by another app (e.g. Teams, Zoom).")
        print(" 2. Try another device index using: python main.py --camera 1")
        detector.close()
        sys.exit(1)

    window_name = "Hand-Gesture PowerPoint Controller (Whole-Hand)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    show_landmarks = True
    prev_time = time.time()
    fps = 30.0

    print("\n[Running] System active! Move your entire hand left or right to swipe slides.")
    print("Press 'Q' inside the camera window to exit.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0.001:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Mirror frame if enabled (makes physical right equal to screen right)
            if cfg.camera.mirror_feed:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # 1. Detect Hand & 21 Landmarks
            hand_data = detector.detect(frame)

            # 2. Update Whole-Hand Motion Tracker
            motion_status = motion_tracker.update(hand_data, current_time, w, h)

            # 3. Process Trajectory & Detect Swipe
            history = motion_tracker.get_history()
            gesture = swipe_recognizer.process(history, current_time)

            if gesture != Gesture.NONE:
                # Trigger PowerPoint Slide Navigation
                presentation_controller.handle_gesture(gesture)

                # Trigger Visual Banner on UI
                ui_renderer.show_swipe_banner(gesture)

                # Reset motion buffer immediately so return motion cannot trigger
                motion_tracker.reset()

            # 4. Render UI Visualizations
            display_frame = ui_renderer.render(
                frame=frame,
                hand_data=hand_data,
                motion_tracker=motion_tracker,
                swipe_recognizer=swipe_recognizer,
                presentation_controller=presentation_controller,
                fps=fps,
                show_landmarks=show_landmarks
            )

            cv2.imshow(window_name, display_frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("[Shutdown] User requested exit.")
                break
            elif key in (ord('c'), ord('C')):
                presentation_controller.toggle_enabled()
            elif key in (ord('m'), ord('M')):
                cfg.camera.mirror_feed = not cfg.camera.mirror_feed
                print(f"[Setting] Camera mirror: {'ENABLED' if cfg.camera.mirror_feed else 'DISABLED'}")
            elif key in (ord('r'), ord('R')):
                motion_tracker.reset()
                swipe_recognizer.reset_cooldown()
                print("[Reset] Motion buffer and cooldown reset.")
            elif key in (ord('h'), ord('H')):
                show_landmarks = not show_landmarks
                print(f"[Setting] Landmarks display: {'VISIBLE' if show_landmarks else 'HIDDEN'}")

    except KeyboardInterrupt:
        print("\n[Shutdown] Terminated by user interrupt.")
    finally:
        print("[Cleanup] Releasing camera and detector...")
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("[Shutdown] Complete. Goodbye!")


if __name__ == "__main__":
    run_app()

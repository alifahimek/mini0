"""
Presentation Controller Module.
Dispatches keyboard commands via PyAutoGUI to control Microsoft PowerPoint,
Google Slides, PDF presentations, or Keynote.
"""

import time
from typing import Callable, List, Optional
import pyautogui

from config import KeyConfig
from gesture_recognizer import Gesture

# Ensure PyAutoGUI safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02


class PresentationController:
    """
    Translates detected gestures into PowerPoint keyboard navigation commands.
    Right Swipe -> Right Arrow (Next slide)
    Left Swipe  -> Left Arrow  (Previous slide)
    """

    def __init__(self, config: KeyConfig):
        self.config = config
        self.enabled: bool = config.enable_key_emission

        # Statistics and history
        self.next_slide_count: int = 0
        self.prev_slide_count: int = 0
        self.last_action_time: float = 0.0
        self.last_action_name: str = "None"
        self.event_log: List[str] = []

        # Optional listener callback
        self.on_slide_change: Optional[Callable[[str], None]] = None

    def handle_gesture(self, gesture: Gesture) -> bool:
        """
        Processes a detected gesture. Returns True if a slide action was executed.
        """
        if gesture == Gesture.RIGHT_SWIPE:
            return self.next_slide()
        elif gesture == Gesture.LEFT_SWIPE:
            return self.prev_slide()
        return False

    def next_slide(self) -> bool:
        """Dispatches Next Slide command (Right Arrow key)."""
        action = f"NEXT SLIDE [{self.config.next_slide_key.upper()}]"
        self.last_action_name = action
        self.last_action_time = time.time()
        self.next_slide_count += 1
        self._log_event(action)

        if self.enabled:
            try:
                pyautogui.press(self.config.next_slide_key)
            except Exception as e:
                print(f"[PresentationController] Error pressing '{self.config.next_slide_key}': {e}")
                return False

        if self.on_slide_change:
            self.on_slide_change(action)

        return True

    def prev_slide(self) -> bool:
        """Dispatches Previous Slide command (Left Arrow key)."""
        action = f"PREV SLIDE [{self.config.prev_slide_key.upper()}]"
        self.last_action_name = action
        self.last_action_time = time.time()
        self.prev_slide_count += 1
        self._log_event(action)

        if self.enabled:
            try:
                pyautogui.press(self.config.prev_slide_key)
            except Exception as e:
                print(f"[PresentationController] Error pressing '{self.config.prev_slide_key}': {e}")
                return False

        if self.on_slide_change:
            self.on_slide_change(action)

        return True

    def toggle_enabled(self) -> bool:
        """Toggles keyboard emission on/off."""
        self.enabled = not self.enabled
        state = "ENABLED" if self.enabled else "MUTED (Testing Only)"
        print(f"[PresentationController] Key emission is now: {state}")
        return self.enabled

    def _log_event(self, text: str):
        timestamp_str = time.strftime("%H:%M:%S")
        entry = f"[{timestamp_str}] {text} (Total: Next={self.next_slide_count}, Prev={self.prev_slide_count})"
        self.event_log.append(entry)
        if len(self.event_log) > 50:
            self.event_log.pop(0)
        print(f"[PresentationController] {entry}")

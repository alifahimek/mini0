"""
Interactive Demo Presentation Deck.
A lightweight, standalone slide viewer built with Tkinter.
Responds in real-time to the PyAutoGUI Right and Left arrow keys dispatched by main.py.
Use this to test your hand gestures immediately without needing Microsoft PowerPoint open!
"""

import tkinter as tk
from typing import List, Tuple


SLIDES_DATA = [
    {
        "title": "Whole-Hand Gesture Presentation System",
        "subtitle": "Computer Vision & Motion-Based PowerPoint Navigation",
        "bullet_points": [
            "Controlled using natural hand swipes (MediaPipe + OpenCV + PyAutoGUI)",
            "Pose-Agnostic: Works with any finger pose (open palm, fist, relaxed, loose fingers)",
            "Rotation-Agnostic: Palm-facing, back-facing, tilted, or sideways",
            "Try it now: Swipe your hand to the RIGHT to move to Slide 2!"
        ],
        "color_accent": "#00ADB5",
        "footer": "Slide 1 of 5  |  Waiting for Right Swipe (Next Slide)..."
    },
    {
        "title": "Design Principle: Entire Hand Input",
        "subtitle": "No finger-counting, no pinch gestures, no static pose classifications",
        "bullet_points": [
            "Single fingertips jitter when fingers curl or rotate.",
            "This system tracks the rigid anatomical base: Wrist (0) + MCP Knuckles (1, 5, 9, 13, 17).",
            "Stable whole-hand center remains consistent regardless of orientation.",
            "Swipe RIGHT for Next Slide  |  Swipe LEFT for Previous Slide"
        ],
        "color_accent": "#FF5722",
        "footer": "Slide 2 of 5  |  Use Left/Right swipes to navigate"
    },
    {
        "title": "Robust Across All Hand Orientations",
        "subtitle": "Tested across diverse angles and natural presenter movements",
        "bullet_points": [
            "[✓] Palm facing the webcam",
            "[✓] Back of the hand facing the webcam",
            "[✓] Hand rotated 90 degrees or tilted diagonally",
            "[✓] Hand partially turned sideways",
            "[✓] Different wrist angles and distances from laptop"
        ],
        "color_accent": "#4CAF50",
        "footer": "Slide 3 of 5  |  Swipe Right to continue"
    },
    {
        "title": "Intelligent Motion & Debounce Pipeline",
        "subtitle": "Eliminating false positives and accidental slide changes",
        "bullet_points": [
            "Horizontal Dominance Check: Rejects vertical waving, scratching, or nodding.",
            "Velocity & Distance Threshold: Rejects slow hand repositioning and fidgets.",
            "Debounce Cooldown (1.2s): Guarantees one physical swipe changes exactly one slide.",
            "Buffer Flush: Return strokes never trigger accidental reverse slides."
        ],
        "color_accent": "#E91E63",
        "footer": "Slide 4 of 5  |  Swipe Right to continue"
    },
    {
        "title": "Ready for Microsoft PowerPoint & Beyond",
        "subtitle": "Plug-and-play with any standard presentation software",
        "bullet_points": [
            "Microsoft PowerPoint: Full screen slide show (F5)",
            "Google Slides: Presentation mode",
            "Adobe Acrobat / PDF Presentations",
            "Canva, Apple Keynote, and web slide decks",
            "Everything works seamlessly via standard Left / Right arrow keys!"
        ],
        "color_accent": "#9C27B0",
        "footer": "Slide 5 of 5  |  Swipe Left to go backward!"
    }
]


class PresentationViewer:
    """Tkinter slide show window responding to arrow keys."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Demo Presentation Deck - Hand Gesture Controlled")
        self.root.geometry("880x560")
        self.root.configure(bg="#1E1E24")

        self.current_index = 0

        # Build UI layout
        self._setup_widgets()

        # Bind Left and Right arrow keys
        self.root.bind("<Right>", self.on_right_key)
        self.root.bind("<Left>", self.on_left_key)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.render_slide()

    def _setup_widgets(self):
        # Top banner
        self.top_frame = tk.Frame(self.root, bg="#141419", height=50)
        self.top_frame.pack(fill=tk.X, side=tk.TOP)

        self.app_badge = tk.Label(
            self.top_frame,
            text="POWERPOINT DEMO DECK  •  LISTENING FOR ARROW KEYS",
            font=("Segoe UI", 10, "bold"),
            fg="#AAAAAA",
            bg="#141419"
        )
        self.app_badge.pack(side=tk.LEFT, padx=20, pady=12)

        self.counter_label = tk.Label(
            self.top_frame,
            text="1 / 5",
            font=("Segoe UI", 11, "bold"),
            fg="#00ADB5",
            bg="#141419"
        )
        self.counter_label.pack(side=tk.RIGHT, padx=20, pady=12)

        # Main slide card
        self.card = tk.Frame(self.root, bg="#2A2A33", bd=0, relief=tk.FLAT)
        self.card.pack(fill=tk.BOTH, expand=True, padx=40, pady=25)

        # Slide title
        self.title_label = tk.Label(
            self.card,
            text="",
            font=("Segoe UI", 24, "bold"),
            fg="#FFFFFF",
            bg="#2A2A33",
            wraplength=780,
            justify=tk.LEFT
        )
        self.title_label.pack(anchor="w", padx=35, pady=(35, 8))

        # Accent line
        self.accent_canvas = tk.Canvas(self.card, height=4, bg="#2A2A33", highlightthickness=0)
        self.accent_canvas.pack(fill=tk.X, padx=35, pady=(0, 15))

        # Slide subtitle
        self.subtitle_label = tk.Label(
            self.card,
            text="",
            font=("Segoe UI", 13, "italic"),
            fg="#CCCCCC",
            bg="#2A2A33",
            wraplength=780,
            justify=tk.LEFT
        )
        self.subtitle_label.pack(anchor="w", padx=35, pady=(0, 20))

        # Bullets container
        self.bullets_frame = tk.Frame(self.card, bg="#2A2A33")
        self.bullets_frame.pack(fill=tk.BOTH, expand=True, padx=35, pady=(0, 15))

        # Bottom footer bar
        self.footer_label = tk.Label(
            self.root,
            text="",
            font=("Segoe UI", 10),
            fg="#888888",
            bg="#1E1E24"
        )
        self.footer_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 12))

    def render_slide(self):
        slide = SLIDES_DATA[self.current_index]

        self.title_label.config(text=slide["title"])
        self.subtitle_label.config(text=slide["subtitle"])
        self.counter_label.config(
            text=f"Slide {self.current_index + 1} / {len(SLIDES_DATA)}",
            fg=slide["color_accent"]
        )
        self.footer_label.config(text=slide["footer"])

        # Redraw accent line
        self.accent_canvas.delete("all")
        self.accent_canvas.create_rectangle(0, 0, 780, 4, fill=slide["color_accent"], outline="")

        # Clear old bullet points
        for widget in self.bullets_frame.winfo_children():
            widget.destroy()

        # Render bullet points
        for pt in slide["bullet_points"]:
            row = tk.Frame(self.bullets_frame, bg="#2A2A33")
            row.pack(fill=tk.X, anchor="w", pady=5)

            bullet_icon = tk.Label(
                row,
                text="●",
                font=("Segoe UI", 10),
                fg=slide["color_accent"],
                bg="#2A2A33"
            )
            bullet_icon.pack(side=tk.LEFT, anchor="n", padx=(0, 10), pady=2)

            pt_label = tk.Label(
                row,
                text=pt,
                font=("Segoe UI", 12),
                fg="#E0E0E0",
                bg="#2A2A33",
                wraplength=720,
                justify=tk.LEFT
            )
            pt_label.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")

    def on_right_key(self, event=None):
        if self.current_index < len(SLIDES_DATA) - 1:
            self.current_index += 1
            self.render_slide()

    def on_left_key(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.render_slide()


def main():
    root = tk.Tk()
    app = PresentationViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

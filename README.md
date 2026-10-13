# Hand-Gesture-Controlled PowerPoint Presentation System (Closed-Fist Signal)

A Python-based, real-time presentation controller powered by **OpenCV**, **MediaPipe Hands**, and **PyAutoGUI**.

Control slide decks in **Microsoft PowerPoint**, **Google Slides**, **PDF Presenters**, or **Keynote** using **Closed-Fist Swipes**.

---

## ✊ Hand Signal: Closed Fist Requirement (Presenter Clutch)

To eliminate false triggers when speaking and gesturing with open hands:

> **"Slide changes ONLY occur when the hand is formed into a CLOSED FIST."**

### How It Works:
- **Safety Lock (Open Hand):** Gesturing, talking with hands, or pointing with open fingers is **ignored**. Swipes will NOT trigger slide changes.
- **Presenter Clutch (Closed Fist):** Curl your fingers into a closed fist to unlock slide controls:
  - **Closed Fist Swipe RIGHT** $\rightarrow$ **Next Slide** (`'right'` arrow key)
  - **Closed Fist Swipe LEFT** $\rightarrow$ **Previous Slide** (`'left'` arrow key)
- **Robust Across All Fist Orientations:**
  - Fist with palm facing camera
  - Fist with back of hand facing camera (knuckles forward)
  - Fist rotated sideways or tilted diagonally
  - Different wrist and hand angles
- **Toggle Anytime:** Press **`F`** in the camera window or pass `--no-fist-lock` to allow swiping with any hand pose.

---

## 🔄 Processing Pipeline

```
          Webcam Hardware
                 ↓
     ThreadedCamera (Non-blocking background thread @ 30-60+ FPS)
                 ↓
     MediaPipe Hand Detector (Dual-Engine: Tasks v1.0+ & Solutions v0.10.x, confidence: 0.35)
                 ↓
        21 3D Hand Landmarks
                 ↓
      Whole-Hand Palm Anchor Calculation (Wrist + 5 MCP Knuckles)
                 ↓
    Exponential Moving Average (EMA) Smoothing & Sliding Trajectory Window
    (with Grace Frame bridge for fast motions)
                 ↓
     Dual-Strategy Swipe Detection & Rejection Filters:
       • Real-time Stroke Anchor Analysis (Instant trigger)
       • Extrema Pair Analysis (Deceleration confirmation)
       • Natural Arc Tolerance (|ΔX| >= 1.15 * |ΔY|)
       • Biomechanically Tuned for both Inward (Left) and Outward (Right) Swipes
                 ↓
        Snappy Cooldown / Debounce Timer (0.9s) & Buffer Flush
                 ↓
      PyAutoGUI Keyboard Dispatch
       • Right Swipe → 'right' arrow key → Next PowerPoint Slide
       • Left Swipe  → 'left' arrow key  → Previous PowerPoint Slide
                 ↓
   Microsoft PowerPoint / Google Slides / PDF Viewer
```

---

## 📁 Project Structure

```
mini0/
├── camera_stream.py          # High-speed non-blocking threaded camera capture
├── config.py                 # Centralized configuration (thresholds, timings, colors, camera)
├── hand_detector.py          # Unified MediaPipe detector (Tasks API & Legacy Solutions API)
├── motion_tracker.py         # Hand center extraction, EMA filter, sliding window, grace frames
├── gesture_recognizer.py     # Dual-strategy swipe detection (inward/outward tuned, debounce)
├── presentation_controller.py# PyAutoGUI slide dispatcher with safety toggles and event logging
├── ui_renderer.py            # Custom OpenCV HUD, 21-point skeleton, motion trail, cooldown bar
├── main.py                   # Main real-time application runner with CLI arguments
├── demo_slides.py            # Interactive Tkinter presentation viewer for instant testing
├── test_system.py            # Automated test suite (11 unit tests passing)
├── requirements.txt          # Python dependencies
└── hand_landmarker.task      # MediaPipe float16 HandLandmarker model asset
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Controller
```bash
python main.py
```

### 3. Test with the Built-in Demo Slide Deck (Optional)
To test gestures without opening Microsoft PowerPoint:
1. Open a second terminal and run:
   ```bash
   python demo_slides.py
   ```
2. Keep the Demo Deck window focused alongside your webcam window.
3. Swipe your hand to the **Right** $\rightarrow$ Advances to next slide!
4. Swipe your hand to the **Left** $\rightarrow$ Moves back to previous slide!

### 4. Use with Microsoft PowerPoint
1. Start `python main.py`.
2. Open your presentation in **Microsoft PowerPoint** (or Google Slides / Adobe Acrobat).
3. Enter Full Screen Slide Show mode (`F5` in PowerPoint).
4. Swipe naturally in front of your laptop webcam to navigate your slides hands-free!

---

## 🎮 Interactive Keyboard Controls (Inside Camera Window)

| Key | Action |
|:---:|:---|
| **`Q`** or **`ESC`** | Quit application cleanly |
| **`C`** | Toggle PowerPoint key emission (**ACTIVE** vs **MUTED**) |
| **`F`** | Toggle Closed Fist lock requirement (**REQUIRED** vs **ANY POSE**) |
| **`M`** | Toggle camera mirroring (Natural mirror mode ON/OFF) |
| **`R`** | Reset trajectory buffer and cooldown timer |
| **`H`** | Toggle hand landmarks / skeleton visibility |

---

## ⚙️ Command-Line Arguments

You can customize camera settings directly from the terminal:

```bash
python main.py --help
```

| Argument | Default | Description |
|---|:---:|---|
| `--camera` | `0` | Webcam index (change to `1` if using an external USB webcam) |
| `--width` | `640` | Video frame width in pixels |
| `--height` | `480` | Video frame height in pixels |
| `--no-mirror` | `False` | Disable horizontal video mirroring |
| `--no-fist-lock` | `False` | Disable closed-fist requirement (allows any hand pose) |
| `--mute` | `False` | Start with PyAutoGUI key emissions disabled (testing mode) |
| `--cooldown` | `0.9` | Debounce duration between slides in seconds |
| `--model-path` | `None` | Custom path to `hand_landmarker.task` |

Example:
```bash
python main.py --camera 0 --cooldown 1.0
```

---

## 🛡️ False-Positive Rejection Rules

To avoid accidental slide changes while talking or gesturing, the system employs refined multi-factor filters:

1. **Natural Arc Tolerance ($|\Delta X| \ge 1.15 \times |\Delta Y|$):**
   Accommodates natural presenter arm curvature (especially inward left swipes) while firmly rejecting vertical hand waving, scratching your nose, adjusting glasses, nodding, or raising your hand.
2. **Effortless Displacement Threshold ($\ge 12\%$ of frame width, ~77px):**
   Allows natural, comfortable hand flicks rather than exaggerated full-screen arm sweeps, while still rejecting small finger twitches or resting hand jitter.
3. **Natural Velocity Threshold ($\ge 0.28\text{ frame-widths / sec}$):**
   Captures both swift flicks and smooth presentation swipes, while rejecting slow drifts.
4. **Directional Monotonicity ($\ge 55\%$ of trajectory steps in swipe direction):**
   Rejects trembling, waving back and forth, or erratic shaking.
5. **Snappy Cooldown (0.9s) & Instant Buffer Flush:**
   Once a swipe triggers, the motion buffer is instantly cleared, guaranteeing return-stroke immunity (bringing your hand back to rest will never trigger an accidental slide change).

---

## 🧪 Automated Testing

Run the comprehensive unit test suite:
```bash
python test_system.py
```

Tests verify:
- Whole-hand palm center invariance across open vs fist poses.
- Pose stability under $0^\circ, 45^\circ, 90^\circ, 180^\circ, 270^\circ$ hand rotations.
- Left-to-right swipe detection (`RIGHT_SWIPE`).
- Right-to-left swipe detection (`LEFT_SWIPE`).
- Rejection of vertical waving motions.
- Rejection of small accidental jitters.
- Cooldown debounce suppression and return-stroke immunity.
- PyAutoGUI dispatch counts and safety controls.

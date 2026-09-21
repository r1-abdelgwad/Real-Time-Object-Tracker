# Real-Time Object Tracker (CSRT + ORB Re-Detection)

A real-time single-object tracking and automatic recovery application built with Python and OpenCV using a **Hybrid CSRT + ORB Re-Detection** architecture. The user selects a target object using a mouse-drawn bounding box, and the system tracks it continuously in real time. If the object is temporarily lost due to rotation, partial occlusion, motion blur, or leaving the frame, the system automatically recovers it using **ORB feature matching** without requiring user intervention.

---

## Features

- **Real-Time Webcam Tracking**: Smooth 25–30 FPS tracking using OpenCV CSRT.
- **Mouse ROI Selection**: User selects target object interactively (`cv2.selectROI`).
- **ORB Feature Re-Detection**: Automatically extracts and matches ORB (Oriented FAST and Rotated BRIEF) keypoints to re-identify the selected target object.
- **Automatic Recovery**: Re-initializes CSRT upon successful ORB feature recovery.
- **Rotation Robustness**: ORB descriptors provide rotation-invariant matching ($0^\circ \rightarrow 45^\circ \rightarrow 90^\circ \rightarrow 180^\circ$).
- **Temporary Occlusion Tolerance**: Recovers object when hand/obstacle uncovers the target object.
- **Visual State Machine**: HUD clearly displays states:
  - `Tracking: ON` (Green Bounding Box)
  - `Tracking: RECOVERING (Matches: N)` (Amber Status)
  - `Tracking: LOST` (Red Notice)
- **On-Screen FPS Counter**: Real-time frame rate monitoring.
- **Manual Reselection & Reset**: Press **R** to select a new object and refresh reference features.
- **Clean Exit**: Press **Q** or **ESC**.

---

## Technologies

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Programming language |
| OpenCV (opencv-contrib-python) | 4.5+ | CSRT tracking, ORB feature detection, BFMatcher, drawing |
| NumPy | 1.21+ | Matrix/array calculations & point transformations |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/r1-abdelgwad/Real-Time-Object-Tracker.git
cd Real-Time-Object-Tracker
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Important:** Install `opencv-contrib-python`, **not** plain `opencv-python`, as CSRT resides in the contrib module.

---

## How to Run

```bash
python src/tracker.py
```

---

## How to Use

1. **Launch** the app — webcam opens immediately.
2. **Press R** — frame freezes for ROI selection.
3. **Draw a box** around the target object using the mouse.
4. **Press ENTER or SPACE** to confirm.
5. The system starts tracking:
   - **Green Box (`Tracking: ON`)**: Confident real-time CSRT tracking.
   - **Amber Status (`Tracking: RECOVERING`)**: CSRT lost target (active rotation/occlusion); ORB re-detection searching for target.
   - **Red Message (`Tracking Lost`)**: Recovery window timed out (`MAX_FAILURES = 35` frames).
6. **Press R** at any time to re-select a new target.
7. **Press Q or ESC** to exit cleanly.

---

## Architecture & Implementation Details

### Pipeline

```
Webcam Stream
  │
  ├─► User selects ROI (cv2.selectROI)
  ├─► Initialize CSRT Tracker
  ├─► Extract reference ORB descriptors (cv2.ORB_create)
  │
  └─► Main Loop:
        ├─► State: TRACKING
        │      └─► tracker.update(frame)
        │            ├─► success == True  ──► Draw Green Box & "Tracking: ON"
        │            └─► success == False ──► State -> RECOVERING (fail_count = 1)
        │
        ├─► State: RECOVERING (ORB Re-Detection)
        │      ├─► Extract ORB features from current frame
        │      ├─► Match against reference descriptors using BFMatcher + Ratio Test
        │      ├─► Estimate candidate bounding box (Homography RANSAC / Keypoint centroid)
        │      ├─► Validate bounding box (scale, dimensions, boundary limits)
        │      │     ├─► Valid Match  ──► Re-init fresh CSRT tracker -> State: TRACKING
        │      │     └─► Invalid Match ──► fail_count += 1
        │      └─► If fail_count >= MAX_FAILURES ──► State: LOST
        │
        └─► State: LOST
               └─► Display "Tracking Lost - Press R to reselect"
```

### Why CSRT + ORB?

- **CSRT (Continuous Tracking)**: Provides fast, smooth frame-to-frame tracking (~30 FPS) without requiring heavy feature extraction on every frame.
- **ORB (Re-Detection & Recovery)**: Operates when CSRT fails. ORB keypoints and descriptors are **rotation-invariant**, allowing the system to re-identify the exact same object after rotation or brief occlusion, estimate its new position, and re-initialize CSRT.

---

## Limitations

- **Low-Texture Objects**: Objects with uniform colors (e.g. plain blank paper or smooth white mugs) yield few ORB keypoints, making feature-based recovery fallback to manual re-selection (`R`).
- **Extreme Motion Blur**: Rapid camera movement blurs edges, degrading both CSRT correlation and ORB feature matching.
- **Long-Term Disappearance**: If the object leaves the frame for longer than `MAX_FAILURES` (35 frames / ~1.2s), the state transitions to `LOST`.
- **Cluttered Backgrounds**: Objects with identical patterns nearby may produce candidate matches requiring careful ROI selection.

---

## Testing Matrix

| Test Case | Expected Behavior | Result / Status |
|---|---|---|
| **Normal Movement** | Smooth green bounding box tracking (~30 FPS) | Passed ✅ |
| **2D/3D Rotation** | CSRT drops briefly, ORB re-detects target, re-inits CSRT | Passed ✅ |
| **Partial Occlusion** | CSRT maintains track or recovers via ORB | Passed ✅ |
| **Temporary Full Occlusion** | Enters `RECOVERING`, re-acquires target upon un-covering | Passed ✅ |
| **Object Re-entering Frame** | Recovers object if re-entered within 35 frames | Passed ✅ |
| **Manual Reselection (R)** | Clears state & reference features, initializes fresh CSRT | Passed ✅ |

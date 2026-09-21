# Real-Time Object Tracker (CSRT + Dual-Path Re-ID)

A real-time single-object tracking and automatic re-identification application built with Python and OpenCV using a **Hybrid CSRT + Dual-Path Re-ID (ORB + Multi-Scale Template Matching + HSV Color Histogram)** architecture. The user selects a target object using a mouse-drawn bounding box, and the system tracks it continuously in real time with a persistent Target ID. If the object is temporarily lost due to rotation, occlusion, fast motion, or leaving the frame, the system automatically recovers it without requiring manual re-selection.

---

## Features

- **Real-Time Webcam Tracking**: Smooth 25–30 FPS tracking using OpenCV CSRT with channel reliability.
- **Mouse ROI Selection**: User selects target object interactively (`cv2.selectROI`).
- **Persistent Target ID**: ID assigned once at selection and strictly preserved across re-acquisitions.
- **Dual-Path Automatic Recovery**:
  - **Path A (ORB Feature Matching)**: ORB + BFMatcher (Hamming) + Lowe's Ratio Test + RANSAC Homography. Rotation-invariant and precise for textured objects.
  - **Path B (Multi-Scale Template Fallback)**: Multi-scale template matching across dynamic template history for uniform/low-texture objects.
- **HSV Color Histogram Gating**: 2D Hue-Saturation normalized histogram correlation verifies candidate patches to eliminate background and false-color matches.
- **Appearance Drift Protection**: Stored template history updates only during confirmed stable tracking, while reference ORB descriptors and HSV histogram remain anchored to original selection.
- **Visual State Machine**:
  - `IDLE`: Waiting for user target selection (Press **R**).
  - `TRACKING`: Confident CSRT tracking with real-time bounding box and Target ID.
  - `RECOVERING`: CSRT lost target; Re-ID scanning full frame using ORB / Template + HSV.
  - `LOST`: Recovery search timed out (`MAX_SEARCH_FRAMES = 60`); prompts user re-selection.
- **On-Screen FPS Counter**: Rolling average frame rate monitoring.
- **Manual Reselection & Reset**: Press **R** to select a new object.
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
  ├─► User selects ROI (cv2.selectROI) -> Assigned Target ID
  ├─► Initialize CSRT Tracker
  ├─► Extract reference features:
  │     ├─► Reference ORB Keypoints & Descriptors (Anchored)
  │     ├─► Reference HSV 2D Histogram (Anchored)
  │     └─► Initial Template History (Dynamic FIFO queue)
  │
  └─► Main Loop:
        ├─► State: TRACKING
        │      └─► tracker.update(frame)
        │            ├─► success == True:
        │            │     - Draw Green Bounding Box + Target ID + State
        │            │     - After 15+ stable frames: safely update template history
        │            │     - Reset failure counters
        │            └─► success == False:
        │                  - csrt_fail_count += 1
        │                  - If csrt_fail_count >= MAX_TRACKING_FAILURES (5):
        │                      State -> RECOVERING
        │
        ├─► State: RECOVERING (Dual-Path Re-ID)
        │      ├─► Path A: ORB Detection + BFMatcher + Lowe's Ratio + RANSAC
        │      │     └─► Candidate found? -> HSV Color Verification
        │      ├─► Path B (Fallback): Multi-scale Template Matching across history
        │      │     └─► Candidate found? -> HSV Color Verification
        │      ├─► Combined confidence >= RECOVERY_THRESHOLD (0.52):
        │      │     - Re-initialize fresh CSRT tracker at recovered location
        │      │     - Preserve exact Target ID
        │      │     - Reset failure & search counters -> State: TRACKING
        │      └─► No match:
        │            - search_frame_count += 1
        │            - If search_frame_count >= MAX_SEARCH_FRAMES (60):
        │                State -> LOST
        │
        └─► State: LOST
               └─► Display "Tracking Lost - Press R to reselect"
```

### Why Hybrid CSRT + Dual-Path Re-ID?

- **CSRT (Continuous Tracking)**: Channel and spatial reliability maps provide robust real-time tracking (~30 FPS) with low drift.
- **Path A: ORB (Rotation & High Texture)**: Provides fast, rotation-invariant keypoint matching and homography verification when objects rotate or undergo perspective distortion.
- **Path B: Template Matching (Low Texture / Uniform)**: Serves as a fallback when objects lack high-frequency texture details where ORB cannot find enough keypoints.
- **HSV Gating**: Eliminates false positive matches by ensuring the color signature strictly matches the original target.
- **Drift Prevention**: Unlike naive online trackers that update features every frame and eventually track the background, our system keeps the reference ORB descriptors and reference color histogram fixed, while dynamically managing a small FIFO of verified templates.

---

## Limitations

- **Extreme Motion Blur**: Rapid camera shakes blur edges, degrading both CSRT correlation and ORB detection.
- **Total Occlusion Beyond Search Window**: If the object is fully occluded or absent for more than `MAX_SEARCH_FRAMES` (60 frames / ~2s), the system transitions to `LOST` to prevent indefinite background scanning.
- **Extreme Lighting Variations**: Dramatic changes between dark shadows and bright illumination may reduce HSV correlation scores.
- **Severe Out-of-Plane 3D Rotation**: When an object rotates 180 degrees showing an unseen side, re-acquisition requires the visible face to return or manual re-selection (`R`).

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

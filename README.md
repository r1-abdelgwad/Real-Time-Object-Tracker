# Real-Time Object Tracker

A real-time single-object tracker built with Python and OpenCV using the CSRT (Channel and Spatial Reliability Tracking) algorithm. The user selects an object in the first frame using a mouse-drawn bounding box, and the tracker follows it across subsequent frames from a live webcam feed.

---

## Features

- Live webcam feed with real-time tracking
- Live webcam feed with real-time tracking
- Mouse-based object selection using `cv2.selectROI`
- **CSRT Tracker**: High-accuracy correlation filter tracker with channel and spatial reliability
- Real-time on-screen **FPS** counter
- Clean visual status: **`Tracking: ON`** (White bounding box) / **`Tracking Lost`** (Red notice)
- Press **R** to re-select target object at any time
- Press **Q** or **ESC** to exit cleanly

---

## Technologies

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Programming language |
| opencv-contrib-python | 4.5+ | Video capture, CSRT tracking algorithm, drawing |
| numpy | 1.21+ | Array matrix representations |

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

> **Important:** Install `opencv-contrib-python`, **not** plain `opencv-python`, as tracking modules reside in the contrib repository.

---

## How to Run

```bash
python src/tracker.py
```

---

## How to Use

1. **Launch** the application — webcam opens immediately.
2. **Press R** — frame freezes for ROI selection.
3. **Draw a box** around the target object using the mouse.
4. **Press ENTER or SPACE** to confirm selection.
5. The tracker follows the object in real time:
   - **White Box (`Tracking: ON`)**: Active tracking.
   - **Red Message (`Tracking Lost`)**: Object lost or left the frame.
6. **Press R** at any time to re-select a new object.
7. **Press Q or ESC** to exit.

---

## Implementation Details

### Pipeline

```
Webcam
  │
  ├─► cap.read()          # Read frame
  ├─► cv2.selectROI()     # User draws initial bounding box
  ├─► tracker.init()      # Initialize CSRT tracker
  └─► Main Loop:
        cap.read()        # Fetch new frame
        tracker.update()  # CSRT updates object coordinates
          ├─► success == True  ──► Draw White Rectangle & "Tracking: ON"
          └─► success == False ──► Display "Tracking Lost - Press R to reselect"
```

### Tracking Algorithm

**CSRT (Channel and Spatial Reliability Tracking)** was selected because:
- Uses spatial reliability maps to filter out non-target background pixels inside the bounding box.
- Handles **scale changes** (object moving closer/farther from camera).
- Runs efficiently at ~25–30 FPS on CPU for real-time video streams.

---

## Limitations

- **Single Object Only**: Tracks one target at a time.
- **Axis-Aligned Bounding Box**: The bounding box remains horizontal ($\theta = 0^\circ$) and does not rotate with the object.
- **Rotation Sensitivity**: Severe 2D/3D rotations alter feature gradient orientations, causing `tracker.update()` to return `False` and triggering `Tracking Lost`.
- **Full Occlusion**: If the target object is completely blocked (e.g. by a hand or obstacle), the tracker loses track.
- **No Automatic Re-detection**: If tracking is lost, global re-detection is not performed; the user must press `R` to reselect the object.

---

## Testing

| Test Case | Expected Behavior | Possible Failure |
|---|---|---|
| Slow horizontal movement | Box follows smoothly | — |
| Fast horizontal movement | Box may lag or lose target | CSRT search window too small |
| Object gets closer (larger) | Box scales up | May drift at extreme zoom |
| Object moves away (smaller) | Box scales down | May lose at very small size |
| Partial occlusion | Continues tracking | Loses if >60% occluded |
| Full occlusion | Tracking Lost shown | Expected failure |
| Object leaves frame | Tracking Lost shown | Expected failure |
| Low lighting | May degrade | Feature loss |
| Similar objects nearby | May drift to wrong object | Appearance confusion |

---

## Demo

See [`demo/demo.mp4`](demo/demo.mp4) for a recorded demonstration.

---

## Future Improvements

- Add automatic re-detection when tracking fails (combining a detector like HOG+SVM with the tracker).
- Support multi-object tracking.
- Add command-line arguments for camera index and tracker type selection.
- Optional GPU acceleration using OpenCV CUDA modules.
- Benchmark comparison between CSRT, KCF, and MOSSE on the same video sequence.

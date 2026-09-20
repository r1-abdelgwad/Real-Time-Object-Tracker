"""
Real-Time Object Tracker using OpenCV CSRT Algorithm
=====================================================
Author  : Rawan Abdelgwad
Date    : 2026
Task    : ML & Computer Vision Internship Assessment

How it works (Pipeline):
  Webcam → First Frame → User selects ROI → Initialize CSRT Tracker
  → Read next frame → tracker.update(frame) → Draw bounding box
  → Display live video → Handle failure → Press R to reselect / Q to quit
"""

import cv2
import time

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
WINDOW_NAME   = "Real-Time Object Tracker"
COLOR_SUCCESS = (255, 255, 255)  # White Bounding Box
COLOR_FAILURE = (0, 0, 255)      # Red Text
COLOR_INFO    = (255, 255, 0)    # Cyan Hints
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
#  TRACKER FACTORY
# ─────────────────────────────────────────────
def create_tracker():
    """Create a CSRT tracker instance supporting OpenCV 4.x and 5.x."""
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, 'TrackerCSRT'):
        return cv2.TrackerCSRT.create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT'):
        return cv2.legacy.TrackerCSRT.create()
    else:
        raise RuntimeError(
            "CSRT tracker not found.\n"
            "Make sure you installed: pip install opencv-contrib-python"
        )


# ─────────────────────────────────────────────
#  CAMERA & FRAME HELPERS
# ─────────────────────────────────────────────
def open_camera(camera_index=0):
    """Open webcam capture."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={camera_index}).")
    return cap


def read_frame(cap):
    """Read a single frame from camera."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from camera.")
    return frame


# ─────────────────────────────────────────────
#  ROI SELECTION
# ─────────────────────────────────────────────
def select_roi(frame):
    """Prompt user to select a bounding box using mouse."""
    display = frame.copy()
    cv2.putText(
        display,
        "Draw box around object -> Press ENTER or SPACE to confirm",
        (10, 30), FONT, 0.6, COLOR_INFO, 2
    )
    bbox = cv2.selectROI(
        "Select Object - Press ENTER/SPACE to confirm, C to cancel",
        display,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow("Select Object - Press ENTER/SPACE to confirm, C to cancel")

    if bbox[2] == 0 or bbox[3] == 0:
        return None
    return bbox


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_bounding_box(frame, bbox, color=COLOR_SUCCESS):
    """Draw bounding rectangle on frame."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)


def draw_label(frame, text, position, color):
    """Draw text label with dark background overlay."""
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, 0.65, 2)
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 4),
        (x + text_w, y + baseline),
        (0, 0, 0),
        cv2.FILLED
    )
    cv2.putText(frame, text, (x, y - 4), FONT, 0.65, color, 2)


def draw_fps(frame, fps):
    """Render FPS counter in top-right corner."""
    text = f"FPS: {fps:.1f}"
    frame_w = frame.shape[1]
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    x = frame_w - text_w - 15
    y = text_h + 15
    cv2.putText(frame, text, (x, y), FONT, 0.65, COLOR_INFO, 2)


def draw_controls(frame):
    """Render user controls hint at bottom of frame."""
    frame_h = frame.shape[0]
    hint = "Q / ESC: Quit    R: Re-select Object"
    cv2.putText(frame, hint, (10, frame_h - 15), FONT, 0.5, (200, 200, 200), 1)


# ─────────────────────────────────────────────
#  MAIN APPLICATION LOOP
# ─────────────────────────────────────────────
def run():
    try:
        cap = open_camera(camera_index=0)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    tracker  = None
    tracking = False

    print("[INFO] Camera opened successfully.")
    print("[INFO] Press R to select an object to track.")
    print("[INFO] Press Q or ESC to quit.")

    while True:
        try:
            frame = read_frame(cap)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            break

        fps = 0.0

        if tracking and tracker is not None:
            t_start = time.time()
            success, bbox = tracker.update(frame)
            t_end = time.time()

            elapsed = t_end - t_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            if success:
                draw_bounding_box(frame, bbox, COLOR_SUCCESS)
                x, y = int(bbox[0]), int(bbox[1])
                draw_label(frame, "Tracking: ON", (x, y - 10), COLOR_SUCCESS)
            else:
                draw_label(frame, "Tracking Lost - Press R to reselect", (10, 80), COLOR_FAILURE)
                tracking = False

        draw_fps(frame, fps)
        draw_controls(frame)

        if not tracking:
            draw_label(frame, "Press R to select object", (10, 40), COLOR_INFO)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:
            print("[INFO] Exiting application...")
            break

        elif key == ord('r'):
            print("[INFO] Initiating object selection...")
            try:
                frame_for_roi = read_frame(cap)
            except RuntimeError:
                frame_for_roi = frame

            bbox = select_roi(frame_for_roi)

            if bbox is None:
                print("[WARNING] Selection cancelled.")
                tracking = False
            else:
                try:
                    tracker = create_tracker()
                    tracker.init(frame_for_roi, bbox)
                    tracking = True
                    print(f"[INFO] CSRT Tracker initialized with bbox: {bbox}")
                except RuntimeError as e:
                    print(f"[ERROR] Tracker initialization failed: {e}")
                    tracking = False

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released cleanly.")


if __name__ == "__main__":
    run()

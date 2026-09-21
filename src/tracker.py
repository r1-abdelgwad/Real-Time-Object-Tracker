"""
Real-Time Object Tracker with CSRT & Re-Identification (Re-ID)
===============================================================
Author  : Rawan Abdelgwad
Date    : 2026

Features:
  - Real-Time CSRT tracking for smooth high-FPS object tracking.
  - ReIDMatcher module integration for ORB feature-based object recovery.
  - State machine: IDLE, TRACKING, RECOVERING, LOST.

How it works (Pipeline):
  Webcam → User selects ROI → CSRT tracks object → If lost, ORB Re-ID
  searches the full frame → If match found, CSRT re-initializes → Repeat.

Controls:
  R       → Select / re-select an object to track
  Q / ESC → Quit the application
"""

import cv2
import time
import numpy as np
import sys
import os

# Enable robust importing from local src directory regardless of working directory
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from reid_matcher import ReIDMatcher
except ImportError:
    try:
        from src.reid_matcher import ReIDMatcher
    except ImportError:
        raise RuntimeError("Could not import reid_matcher module. Ensure src/reid_matcher.py exists.")

# ─────────────────────────────────────────────
#  STATE MACHINE & CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────
STATE_IDLE       = "IDLE"
STATE_TRACKING   = "TRACKING"
STATE_RECOVERING = "RECOVERING"
STATE_LOST       = "LOST"

WINDOW_NAME      = "Real-Time Object Tracker"
COLOR_SUCCESS    = (0, 255, 0)     # Green  → Tracking ON
COLOR_RECOVERING = (0, 191, 255)   # Cyan   → Recovering
COLOR_FAILURE    = (0, 0, 255)     # Red    → Tracking Lost
COLOR_INFO       = (255, 255, 0)   # Yellow → FPS / Hints
BOX_THICKNESS    = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# ── Recovery & Feature Matching Parameters ───
MAX_FAILURES         = 35     # Frames grace window (~1.2s at 30 FPS)
MIN_GOOD_MATCHES     = 6      # Minimum ORB matches required for candidate recovery
RATIO_TEST_THRESHOLD = 0.75   # Lowe's ratio test threshold for feature matching
MAX_ORB_FEATURES     = 500    # Maximum ORB keypoints to detect in reference ROI


# ─────────────────────────────────────────────
#  TRACKER FACTORY
# ─────────────────────────────────────────────
def create_tracker():
    """
    Create a CSRT tracker instance supporting OpenCV 4.x and 5.x.

    CSRT (Channel and Spatial Reliability Tracker) is chosen because:
      - It achieves high accuracy on deformable and rotating objects.
      - It uses channel reliability maps to ignore unreliable colour channels.
      - It is robust to partial occlusion and moderate motion blur.
    """
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
    """
    Open webcam capture with DirectShow fallback for Windows stability.
    """
    # 1. Try DirectShow backend on Windows (fastest & most stable)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # 2. Fallback to default OpenCV camera backend
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # 3. Try secondary camera index 1 if 0 is unavailable
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (index={camera_index}).\n"
            "Check that your webcam is connected and not used by another app."
        )
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
#  MAIN APPLICATION LOOP (HYBRID STATE MACHINE)
# ─────────────────────────────────────────────
def run():
    """
    Main application entry point.

    State machine:
      IDLE       → Waiting for user to select an object (press R).
      TRACKING   → CSRT is actively tracking the selected object.
      RECOVERING → CSRT lost the object; ORB Re-ID scanning every frame.
      LOST       → Recovery window timed out; user must re-select.

    Why does the tracker lose objects?
      - Rotation:   CSRT uses appearance templates that degrade when the
                    object rotates significantly (template mismatch).
      - Occlusion:  When another object fully covers the target, CSRT has
                    no appearance signal to lock on to.
      - Fast motion / blur: The search window may not cover the object's
                    new position, or blur reduces discriminative features.
      The Re-ID module addresses all three by searching the entire frame
      using ORB feature matching rather than a restricted search window.
    """
    try:
        cap = open_camera(camera_index=0)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    tracker         = None
    tracking_state  = STATE_IDLE
    last_valid_bbox = None
    failure_counter = 0

    # Instantiate ReIDMatcher module
    reid = ReIDMatcher(
        max_features=MAX_ORB_FEATURES,
        min_good_matches=MIN_GOOD_MATCHES,
        ratio_threshold=RATIO_TEST_THRESHOLD
    )

    print("[INFO] Camera opened successfully.")
    print("[INFO] Press R to select an object to track.")
    print("[INFO] Press Q or ESC to quit.")

    fps = 0.0

    while True:
        try:
            frame = read_frame(cap)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            break

        t_start = time.time()

        # ── State 1: TRACKING ─────────────────────────────────────────
        if tracking_state == STATE_TRACKING and tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                last_valid_bbox = bbox
                failure_counter = 0
                draw_bounding_box(frame, bbox, COLOR_SUCCESS)
                x, y = int(bbox[0]), int(bbox[1])
                draw_label(frame, "Tracking: ON", (x, max(y - 10, 20)), COLOR_SUCCESS)
            else:
                # CSRT failed → Transition to RECOVERING state
                tracking_state = STATE_RECOVERING
                failure_counter = 1
                print("[INFO] CSRT lost target. Entering RECOVERING state...")

        # ── State 2: RECOVERING (ORB Feature Re-Detection) ────────────
        if tracking_state == STATE_RECOVERING:
            recovered, recovered_bbox, match_count = reid.recover_object(frame)

            if recovered:
                # Re-initialize fresh CSRT tracker on recovered location
                try:
                    tracker = create_tracker()
                    tracker.init(frame, recovered_bbox)
                    tracking_state  = STATE_TRACKING
                    last_valid_bbox = recovered_bbox
                    failure_counter = 0

                    draw_bounding_box(frame, recovered_bbox, COLOR_SUCCESS)
                    x, y = int(recovered_bbox[0]), int(recovered_bbox[1])
                    draw_label(frame, f"Tracking: RECOVERED ({match_count} pts)", (x, max(y - 10, 20)), COLOR_SUCCESS)
                    print(f"[INFO] Target recovered via ORB matching! Matches: {match_count}, bbox: {recovered_bbox}")
                except RuntimeError as e:
                    print(f"[ERROR] CSRT re-initialization failed: {e}")
                    failure_counter += 1
            else:
                failure_counter += 1
                status_text = f"Recovering... ({failure_counter}/{MAX_FAILURES}) | Matches: {match_count}"
                draw_label(frame, status_text, (10, 80), COLOR_RECOVERING)

                # Timeout check → Declare LOST
                if failure_counter >= MAX_FAILURES:
                    tracking_state = STATE_LOST
                    print("[INFO] Recovery window timed out. Target declared LOST.")

        # ── State 3: LOST ─────────────────────────────────────────────
        if tracking_state == STATE_LOST:
            draw_label(frame, "Tracking Lost - Press R to reselect", (10, 80), COLOR_FAILURE)

        # ── FPS Calculation ───────────────────────────────────────────
        t_end   = time.time()
        elapsed = t_end - t_start
        fps     = 1.0 / elapsed if elapsed > 0 else fps

        # Render HUD
        draw_fps(frame, fps)
        draw_controls(frame)

        if tracking_state in [STATE_IDLE, STATE_LOST]:
            draw_label(frame, "Press R to select object", (10, 40), COLOR_INFO)

        cv2.imshow(WINDOW_NAME, frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:  # Q or ESC
            print("[INFO] Exiting application...")
            break

        elif key == ord('r'):      # R → Reselect ROI & Re-extract Features
            print("[INFO] Initiating object selection...")
            try:
                frame_for_roi = read_frame(cap)
            except RuntimeError:
                frame_for_roi = frame

            bbox = select_roi(frame_for_roi)

            if bbox is None:
                print("[WARNING] Selection cancelled.")
                tracking_state = STATE_IDLE
            else:
                try:
                    # 1. Create & initialize CSRT tracker
                    tracker = create_tracker()
                    tracker.init(frame_for_roi, bbox)

                    # 2. Extract ORB reference features for ReID module
                    reid.extract_reference_features(frame_for_roi, bbox)

                    tracking_state  = STATE_TRACKING
                    last_valid_bbox = bbox
                    failure_counter = 0

                    num_kp = len(reid.ref_keypoints) if reid.ref_keypoints is not None else 0
                    print(f"[INFO] CSRT Tracker initialized. ORB Reference Keypoints: {num_kp}")

                except RuntimeError as e:
                    print(f"[ERROR] Tracker initialization failed: {e}")
                    tracking_state = STATE_LOST

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released cleanly.")


if __name__ == "__main__":
    run()

"""
Real-Time Object Tracker with CSRT & Re-Identification (Re-ID)
===============================================================
Author  : Rawan Abdelgwad
Date    : 2026

Pipeline:
  Webcam -> User selects ROI -> CSRT tracks ->
  [if lost] Re-ID module searches full frame ->
  [if found] CSRT re-initialises on recovered location -> repeat.

State Machine:
  IDLE       - Waiting for user to select an object (press R).
  TRACKING   - CSRT is actively tracking. Profile updated every N frames.
  RECOVERING - CSRT failed MAX_TRACKING_FAILURES times; Re-ID scanning.
  LOST       - Re-ID timed out after MAX_SEARCH_FRAMES; needs re-selection.

Re-ID two-path recovery (see reid_matcher.py):
  Path A (ORB)      - Primary;  works best on textured objects.
  Path B (Template) - Fallback; works on featureless / uniform objects.
  Both paths gated by HSV histogram colour check.

Controls:
  R       - Select / re-select an object to track
  Q / ESC - Quit the application
"""

import cv2
import time
import sys
import os

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from reid_matcher import ReIDMatcher
except ImportError:
    try:
        from src.reid_matcher import ReIDMatcher
    except ImportError:
        raise RuntimeError(
            "Cannot import ReIDMatcher. Make sure src/reid_matcher.py exists."
        )

# ─────────────────────────────────────────────────────────────────────────────
#  STATE LABELS
# ─────────────────────────────────────────────────────────────────────────────
STATE_IDLE       = "IDLE"
STATE_TRACKING   = "TRACKING"
STATE_RECOVERING = "RECOVERING"
STATE_LOST       = "LOST"

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Tracking robustness
# How many consecutive CSRT failures trigger RECOVERING mode.
# 5 frames (~0.17 s at 30 FPS) avoids false alarms on single bad frames.
MAX_TRACKING_FAILURES  = 5

# Max frames we spend in RECOVERING before declaring the target LOST.
# 60 frames = ~2 s at 30 FPS — enough for a slow re-entry into frame.
MAX_SEARCH_FRAMES      = 60

# Profile update
# Update stored templates every N successful CSRT frames to adapt to slow
# appearance changes (lighting drift, slight pose change).
PROFILE_UPDATE_INTERVAL = 20

# Minimum consecutive successes before we trust tracking enough to update profile.
MIN_STABLE_FRAMES_FOR_UPDATE = 15

# Display
WINDOW_NAME      = "Real-Time Object Tracker"
COLOR_SUCCESS    = (0, 255,   0)    # Green  - Tracking ON
COLOR_RECOVERING = (0, 191, 255)    # Cyan   - Recovering
COLOR_LOST       = (0,   0, 255)    # Red    - Lost
COLOR_INFO       = (255, 255,  0)   # Yellow - FPS / hints
BOX_THICKNESS    = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# FPS smoothing
FPS_WINDOW = 10   # rolling average over this many frames


# ─────────────────────────────────────────────────────────────────────────────
#  TRACKER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def create_tracker():
    """
    Instantiate a CSRT tracker compatible with OpenCV 4.x and 5.x.

    CSRT is chosen because:
      - Channel reliability maps downweight unreliable colour channels.
      - Spatial reliability focuses on the discriminative foreground region.
      - More robust to partial occlusion and mild rotation than KCF/MOSSE.

    Known limitations addressed by Re-ID module:
      - Large rotation: appearance template degrades -> ORB re-detects.
      - Full occlusion: no signal -> Re-ID waits for target to reappear.
      - Fast motion outside search window -> Re-ID scans the full frame.
    """
    for creator in [
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT.create(),
        lambda: cv2.legacy.TrackerCSRT.create(),
    ]:
        try:
            t = creator()
            if t is not None:
                return t
        except (AttributeError, cv2.error):
            continue

    raise RuntimeError(
        "CSRT tracker not found.\n"
        "Install via:  pip install opencv-contrib-python"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(camera_index=0):
    """Open webcam; tries DirectShow first on Windows for stability."""
    backends = [
        (camera_index, cv2.CAP_DSHOW),
        (camera_index, cv2.CAP_ANY),
        (1,            cv2.CAP_DSHOW),
    ]
    for idx, backend in backends:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            return cap

    raise RuntimeError(
        f"Cannot open camera (index={camera_index}).\n"
        "Check that your webcam is connected and not in use by another app."
    )


def read_frame(cap):
    """Read one frame; raises RuntimeError on failure."""
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to read a frame from the camera.")
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  ROI SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_roi(frame):
    """Show selectROI dialog; returns bbox tuple or None if cancelled."""
    display = frame.copy()
    cv2.putText(
        display,
        "Draw box -> ENTER/SPACE to confirm   C to cancel",
        (10, 30), FONT, 0.6, COLOR_INFO, 2,
    )
    win = "Select Target — ENTER to confirm, C to cancel"
    bbox = cv2.selectROI(win, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)
    return bbox if (bbox[2] > 0 and bbox[3] > 0) else None


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_bounding_box(frame, bbox, color):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)


def draw_label(frame, text, position, color):
    """Text with a solid black backing for legibility."""
    x, y = position
    (tw, th), bl = cv2.getTextSize(text, FONT, 0.62, 2)
    cv2.rectangle(frame, (x, y - th - bl - 4), (x + tw, y + bl),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y - 4), FONT, 0.62, color, 2)


def draw_fps(frame, fps):
    text  = f"FPS: {fps:.1f}"
    fw    = frame.shape[1]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.62, 2)
    cv2.putText(frame, text, (fw - tw - 15, th + 15), FONT, 0.62, COLOR_INFO, 2)


def draw_controls(frame):
    fh = frame.shape[0]
    cv2.putText(frame, "Q/ESC: Quit   R: Re-select",
                (10, fh - 12), FONT, 0.48, (190, 190, 190), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APPLICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run():
    # ── Camera ────────────────────────────────────────────────────────────
    try:
        cap = open_camera(camera_index=0)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    # ── State variables ────────────────────────────────────────────────────
    tracker             = None
    tracking_state      = STATE_IDLE
    last_valid_bbox     = None

    # target_id is assigned once at first selection and NEVER changes on
    # re-acquisition — the tracker always refers to the same physical object.
    target_id           = 0
    next_id             = 1

    # Failure / search counters
    csrt_fail_count     = 0   # consecutive CSRT failures inside TRACKING
    search_frame_count  = 0   # frames spent in RECOVERING

    # Profile update tracking
    consecutive_successes = 0  # consecutive successful CSRT frames

    # FPS rolling average
    fps_times           = []
    fps                 = 0.0

    # ── Re-ID module ───────────────────────────────────────────────────────
    reid = ReIDMatcher()

    print("[INFO] Camera ready.")
    print("[INFO] Press R to select an object, Q/ESC to quit.")

    # ── Main loop ──────────────────────────────────────────────────────────
    while True:
        t0 = time.time()

        try:
            frame = read_frame(cap)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            break

        # ── STATE: TRACKING ───────────────────────────────────────────────
        if tracking_state == STATE_TRACKING and tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                last_valid_bbox     = bbox
                csrt_fail_count     = 0
                consecutive_successes += 1

                # Draw tracker output
                draw_bounding_box(frame, bbox, COLOR_SUCCESS)
                x, y = int(bbox[0]), int(bbox[1])
                draw_label(frame, f"ID:{target_id}  Tracking",
                           (x, max(y - 10, 20)), COLOR_SUCCESS)

                # Update template profile every PROFILE_UPDATE_INTERVAL frames,
                # but only after tracking has been stable long enough.
                if (consecutive_successes >= MIN_STABLE_FRAMES_FOR_UPDATE
                        and consecutive_successes % PROFILE_UPDATE_INTERVAL == 0):
                    reid.update_profile(frame, bbox, confidence=1.0)

            else:
                # Single CSRT failure -- accumulate before switching state
                csrt_fail_count     += 1
                consecutive_successes = 0

                if csrt_fail_count >= MAX_TRACKING_FAILURES:
                    tracking_state     = STATE_RECOVERING
                    search_frame_count = 0
                    csrt_fail_count    = 0
                    print(f"[INFO] CSRT lost target (ID:{target_id}). "
                          "Entering RECOVERING...")

        # ── STATE: RECOVERING ─────────────────────────────────────────────
        if tracking_state == STATE_RECOVERING:
            search_frame_count += 1

            recovered, rec_bbox, confidence, method = reid.recover_object(frame)

            if recovered:
                # Re-initialise CSRT on the recovered location.
                # target_id is PRESERVED -- same object, same ID.
                try:
                    tracker = create_tracker()
                    tracker.init(frame, rec_bbox)

                    tracking_state      = STATE_TRACKING
                    last_valid_bbox     = rec_bbox
                    csrt_fail_count     = 0
                    consecutive_successes = 0   # wait before updating profile again
                    search_frame_count  = 0

                    draw_bounding_box(frame, rec_bbox, COLOR_SUCCESS)
                    x, y = int(rec_bbox[0]), int(rec_bbox[1])
                    draw_label(
                        frame,
                        f"ID:{target_id}  Re-acquired via {method} "
                        f"(conf:{confidence:.2f})",
                        (x, max(y - 10, 20)),
                        COLOR_SUCCESS,
                    )
                    print(f"[INFO] Target ID:{target_id} re-acquired via {method}. "
                          f"Confidence={confidence:.2f}  bbox={rec_bbox}")

                except RuntimeError as e:
                    print(f"[ERROR] CSRT re-init failed: {e}")
                    search_frame_count += 1   # count this attempt as used

            else:
                pct  = int(search_frame_count / MAX_SEARCH_FRAMES * 100)
                draw_label(
                    frame,
                    f"Recovering ID:{target_id} ... {search_frame_count}/{MAX_SEARCH_FRAMES}",
                    (10, 80),
                    COLOR_RECOVERING,
                )

                if search_frame_count >= MAX_SEARCH_FRAMES:
                    tracking_state = STATE_LOST
                    print(f"[INFO] Recovery timed out. "
                          f"Target ID:{target_id} declared LOST.")

        # ── STATE: LOST ───────────────────────────────────────────────────
        if tracking_state == STATE_LOST:
            draw_label(frame, "Target LOST — press R to re-select", (10, 80),
                       COLOR_LOST)

        # ── FPS (rolling average) ─────────────────────────────────────────
        t1 = time.time()
        dt = t1 - t0
        fps_times.append(dt)
        if len(fps_times) > FPS_WINDOW:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

        # ── HUD overlay ───────────────────────────────────────────────────
        draw_fps(frame, fps)
        draw_controls(frame)

        if tracking_state in (STATE_IDLE, STATE_LOST):
            draw_label(frame, "Press R to select object", (10, 40), COLOR_INFO)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):           # Q or ESC -> quit
            print("[INFO] Exiting...")
            break

        elif key == ord('r'):               # R -> select / re-select target
            print("[INFO] Opening ROI selector...")
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
                    # 1. Assign a persistent target ID (only on fresh selection,
                    #    not on re-acquisition — re-acquisition keeps the old ID)
                    target_id = next_id
                    next_id  += 1

                    # 2. Initialise CSRT
                    tracker = create_tracker()
                    tracker.init(frame_for_roi, bbox)

                    # 3. Build Re-ID profile (ORB + HSV + initial template)
                    reid.extract_reference_features(frame_for_roi, bbox)

                    # 4. Reset all counters
                    tracking_state        = STATE_TRACKING
                    last_valid_bbox       = bbox
                    csrt_fail_count       = 0
                    consecutive_successes = 0
                    search_frame_count    = 0

                    n_kp = (len(reid.ref_keypoints)
                            if reid.ref_keypoints is not None else 0)
                    print(f"[INFO] Target ID:{target_id} selected. "
                          f"ORB keypoints={n_kp}  "
                          f"Templates={len(reid.template_history)}")

                except RuntimeError as e:
                    print(f"[ERROR] Initialisation failed: {e}")
                    tracking_state = STATE_LOST

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released.")


if __name__ == "__main__":
    run()

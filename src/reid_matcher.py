"""
Re-Identification (Re-ID) & Feature Matching Module
===================================================
Author  : Rawan Abdelgwad
Date    : 2026

Pipeline (two-stage verification):
  Stage 1 – ORB Feature Matching
    Extracts binary ORB descriptors from the reference ROI and matches them
    against the current frame using BFMatcher + Lowe's ratio test.
    Homography (RANSAC) or centroid estimation locates the candidate bbox.

  Stage 2 – HSV Color Histogram Verification  ← NEW
    After Stage 1 proposes a candidate bbox, its HSV hue-saturation
    histogram is compared against the reference ROI histogram.
    Candidates whose color distribution diverges too far are rejected,
    preventing false recoveries on visually similar but differently
    colored objects (e.g. a red cup vs a green bottle nearby).

Why two stages?
  ORB matches texture/edges → can match wrong objects with similar texture.
  HSV histograms capture color identity → filters colour-similar impostors.
  Together they significantly reduce false-positive recoveries.
"""

import cv2
import numpy as np


class ReIDMatcher:
    # ── Histogram verification threshold ──────────────────────────────
    # Correlation ranges from 0.0 (no match) to 1.0 (perfect match).
    # 0.55 is a balanced threshold: strict enough to reject similar-colour
    # neighbours, lenient enough to tolerate lighting changes.
    HIST_CORR_THRESHOLD = 0.55

    def __init__(self, max_features=500, min_good_matches=6, ratio_threshold=0.75):
        self.max_features      = max_features
        self.min_good_matches  = min_good_matches
        self.ratio_threshold   = ratio_threshold

        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Stage 1: ORB reference data
        self.ref_keypoints   = []
        self.ref_descriptors = None
        self.ref_w           = 0
        self.ref_h           = 0

        # Stage 2: HSV histogram reference  ← NEW
        self.ref_hist = None

    # ─────────────────────────────────────────────────────────────────
    #  REFERENCE EXTRACTION
    # ─────────────────────────────────────────────────────────────────
    def extract_reference_features(self, frame, bbox):
        """
        Extract ORB descriptors AND HSV histogram from the initial ROI.
        Both are stored for two-stage verification during recovery.
        """
        x, y, w, h = [int(v) for v in bbox]
        frame_h, frame_w = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.ref_keypoints, self.ref_descriptors = [], None
            self.ref_hist = None
            self.ref_w, self.ref_h = w, h
            return

        # Stage 1 reference: ORB keypoints + descriptors
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray_roi, None)
        self.ref_w = w
        self.ref_h = h

        # Stage 2 reference: HSV hue-saturation histogram  ← NEW
        self.ref_hist = self._compute_hsv_hist(roi)

    # ─────────────────────────────────────────────────────────────────
    #  HSV HISTOGRAM HELPERS  ← NEW
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_hsv_hist(bgr_roi):
        """
        Compute a normalised 2D hue-saturation histogram from a BGR image patch.

        Using H+S (not V) makes the histogram invariant to brightness changes,
        so the comparison stays valid under different lighting conditions.

        Bins: 30 hue bins × 32 saturation bins = 960 total bins.
        """
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [30, 32],
            [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _verify_color(self, frame, candidate_bbox):
        """
        Stage 2: Compare the candidate region's HSV histogram against the
        reference histogram using correlation metric.

        Returns True only if the color similarity exceeds HIST_CORR_THRESHOLD.
        Returns True unconditionally when no reference histogram exists
        (fallback to ORB-only mode so the tracker still works).
        """
        if self.ref_hist is None:
            return True  # No reference → skip colour check

        x, y, w, h = [int(v) for v in candidate_bbox]
        frame_h, frame_w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            return False

        cand_roi  = frame[y1:y2, x1:x2]
        if cand_roi.size == 0:
            return False

        cand_hist = self._compute_hsv_hist(cand_roi)
        correlation = cv2.compareHist(self.ref_hist, cand_hist, cv2.HISTCMP_CORREL)

        return correlation >= self.HIST_CORR_THRESHOLD

    # ─────────────────────────────────────────────────────────────────
    #  BBOX VALIDATION
    # ─────────────────────────────────────────────────────────────────
    def validate_bbox(self, candidate_bbox, frame_shape):
        """Validate candidate bounding box bounds and scale ratios."""
        if candidate_bbox is None:
            return False

        x, y, w, h = candidate_bbox
        frame_h, frame_w = frame_shape[:2]

        if w < 10 or h < 10:
            return False

        if x < -w or y < -h or x > frame_w or y > frame_h:
            return False

        ref_area   = max(1.0, float(self.ref_w * self.ref_h))
        cand_area  = float(w * h)
        scale_ratio = cand_area / ref_area

        if scale_ratio < 0.2 or scale_ratio > 4.5:
            return False

        return True

    # ─────────────────────────────────────────────────────────────────
    #  MAIN RECOVERY FUNCTION
    # ─────────────────────────────────────────────────────────────────
    def recover_object(self, frame):
        """
        Two-stage object recovery:
          1. ORB feature matching → candidate bounding box.
          2. HSV histogram comparison → colour identity check.

        Returns:
          (recovered: bool, bbox: tuple|None, match_count: int)
        """
        if self.ref_descriptors is None or len(self.ref_descriptors) < 4:
            return False, None, 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.orb.detectAndCompute(gray_frame, None)

        if frame_des is None or len(frame_des) < 2:
            return False, None, 0

        # ── Stage 1: ORB ratio-test matching ──────────────────────────
        matches = self.bf.knnMatch(self.ref_descriptors, frame_des, k=2)

        good_matches = []
        for m_tuple in matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        if match_count < self.min_good_matches:
            return False, None, match_count

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Method A: Homography RANSAC → most accurate
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                pts = np.float32([
                    [0,          0         ],
                    [self.ref_w, 0         ],
                    [self.ref_w, self.ref_h],
                    [0,          self.ref_h]
                ]).reshape(-1, 1, 2)
                dst_corners = cv2.perspectiveTransform(pts, H)

                x_min = float(np.min(dst_corners[:, 0, 0]))
                y_min = float(np.min(dst_corners[:, 0, 1]))
                x_max = float(np.max(dst_corners[:, 0, 0]))
                y_max = float(np.max(dst_corners[:, 0, 1]))

                cand_bbox = (int(x_min), int(y_min),
                             int(x_max - x_min), int(y_max - y_min))

                # ── Stage 2: Colour histogram gate ──────────────────────
                if self.validate_bbox(cand_bbox, frame.shape):
                    if self._verify_color(frame, cand_bbox):
                        return True, cand_bbox, match_count
                    else:
                        # Colour mismatch → reject this candidate silently
                        return False, None, match_count
        except cv2.error:
            pass

        # Method B: Centroid estimation → fallback when homography fails
        dst_coords = dst_pts.reshape(-1, 2)
        center_x   = float(np.median(dst_coords[:, 0]))
        center_y   = float(np.median(dst_coords[:, 1]))

        cand_x    = int(center_x - self.ref_w / 2.0)
        cand_y    = int(center_y - self.ref_h / 2.0)
        cand_bbox = (cand_x, cand_y, int(self.ref_w), int(self.ref_h))

        # ── Stage 2: Colour histogram gate (centroid fallback) ──────────
        if self.validate_bbox(cand_bbox, frame.shape):
            if self._verify_color(frame, cand_bbox):
                return True, cand_bbox, match_count

        return False, None, match_count

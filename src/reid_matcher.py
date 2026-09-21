"""
Re-Identification (Re-ID) Module for Real-Time Object Tracking
==============================================================
Author : Rawan Abdelgwad
Date   : 2026

Architecture:
-------------
                      TARGET SELECTED
                            │
                            ▼
                   Build Target Profile
                            │
               ┌────────────┼────────────┐
               │            │            │
               ▼            ▼            ▼
            ORB         Template       HSV
         descriptors     history      histogram
               │            │            │
               └────────────┼────────────┘
                            │
                            ▼
                         CSRT
                            │
                      Tracking...
                            │
                      CSRT Lost
                            │
                            ▼
                       RECOVERING
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
           ORB Matching          Template Matching
                │                       │
                └───────────┬───────────┘
                            ▼
                      HSV Verification
                            │
                            ▼
                     Geometric Check
                      (if ORB)
                            │
                            ▼
                    Combined Confidence
                            │
                ┌───────────┴───────────┐
                │                       │
             Strong                   Weak
                │                       │
                ▼                       ▼
          RE-ACQUIRED                  Keep
          same ID                     searching
"""

import cv2
import numpy as np


class ReIDMatcher:
    """
    Two-path visual re-identification module:
      - Reference memory: ORB keypoints & descriptors, original template,
        template history FIFO, and 2D HSV color histogram.
      - Recovery pipeline: ORB matching (ratio test + RANSAC homography),
        multi-scale template matching fallback, HSV color gating,
        geometric candidate validation, and combined confidence scoring.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Tunable Parameters & Thresholds
    # ─────────────────────────────────────────────────────────────────────────
    ORB_FEATURES            = 500    # Maximum ORB features to detect
    MIN_ORB_MATCHES         = 8      # Minimum good matches for homography
    RATIO_THRESHOLD         = 0.72   # Lowe's ratio test (m.dist < 0.72 * n.dist)
    MIN_INLIER_RATIO        = 0.30   # Minimum RANSAC inlier ratio
    HSV_CORR_THRESHOLD      = 0.50   # Minimum HSV histogram correlation
    TEMPLATE_THRESHOLD      = 0.50   # Minimum template correlation score
    RECOVERY_THRESHOLD      = 0.52   # Threshold for strong combined confidence
    MAX_TEMPLATES           = 5      # Maximum templates stored in history (FIFO)
    PROFILE_UPDATE_MIN_CONF = 0.75   # Minimum confidence to append new template
    TEMPLATE_SCALES         = [0.75, 0.875, 1.0, 1.125, 1.25]  # Multi-scale ratios

    # ─────────────────────────────────────────────────────────────────────────
    # Constructor
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        # Feature extractors and matchers
        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

        # ── Reference Profile (Anchored Identity) ──
        self.ref_keypoints     = None   # list[cv2.KeyPoint]
        self.ref_descriptors   = None   # np.ndarray of shape (N, 32), uint8
        self.original_template = None   # Initial BGR crop of target
        self.template_history  = []     # Dynamic FIFO list of BGR templates
        self.ref_hist          = None   # Normalized 2D H-S histogram
        self.ref_w             = 0      # Original bounding box width
        self.ref_h             = 0      # Original bounding box height

    # ─────────────────────────────────────────────────────────────────────────
    # Target Profile Construction (On ROI Selection)
    # ─────────────────────────────────────────────────────────────────────────
    def extract_reference_features(self, frame, bbox):
        """
        Builds the initial visual memory profile from the selected ROI.
        Extracts immutable anchors (ORB descriptors & HSV histogram)
        and seeds the template history.
        """
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        self.ref_w = x2 - x1
        self.ref_h = y2 - y1

        # 1. Original Template & Template History
        self.original_template = roi.copy()
        self.template_history  = [self.original_template.copy()]

        # 2. ORB Descriptors & Keypoints
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, None)

        # 3. HSV Color Histogram (2D Hue-Saturation, illumination-invariant)
        self.ref_hist = self._compute_hsv_hist(roi)

    # ─────────────────────────────────────────────────────────────────────────
    # Profile Update (Appearance Drift Protection)
    # ─────────────────────────────────────────────────────────────────────────
    def update_profile(self, frame, bbox, confidence=1.0):
        """
        Safely appends a new appearance template to template_history during
        confirmed stable tracking.

        Drift Protection:
          - Only updates if confidence >= PROFILE_UPDATE_MIN_CONF.
          - Never overwrites the original reference ORB descriptors or HSV hist.
          - Maintains a maximum of MAX_TEMPLATES in a FIFO queue.
        """
        if confidence < self.PROFILE_UPDATE_MIN_CONF:
            return

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return

        self.template_history.insert(0, patch.copy())
        if len(self.template_history) > self.MAX_TEMPLATES:
            self.template_history.pop()

    # ─────────────────────────────────────────────────────────────────────────
    # Recovery Entry Point
    # ─────────────────────────────────────────────────────────────────────────
    def recover_object(self, frame):
        """
        Searches the current frame for the lost target.

        Recovery Steps:
          1. ORB Feature Matching + Lowe's Ratio Test.
          2. Geometric Check via RANSAC Homography.
          3. Multi-scale Template Matching fallback.
          4. HSV Color Histogram Verification.
          5. Bounding box Candidate Validation.
          6. Combined Confidence scoring (Strong vs. Weak).

        Returns:
          recovered  (bool)       : True if a strong candidate is verified
          bbox       (tuple|None) : (x, y, w, h) of recovered target
          confidence (float)      : Combined confidence score in [0.0, 1.0]
          method     (str)        : 'ORB' | 'Template' | 'None'
        """
        fh, fw = frame.shape[:2]

        # ── 1. Path A: ORB Matching + Geometric Check ────────────────────────
        orb_bbox, orb_score = self._match_orb(frame)
        if orb_bbox is not None and self.validate_bbox(orb_bbox, (fh, fw)):
            # HSV Verification on candidate
            hsv_score = self._verify_hsv(frame, orb_bbox)
            if hsv_score >= self.HSV_CORR_THRESHOLD:
                # Combined confidence for ORB
                confidence = 0.65 * orb_score + 0.35 * hsv_score
                if confidence >= self.RECOVERY_THRESHOLD:
                    return True, orb_bbox, float(confidence), "ORB"

        # ── 2. Path B: Multi-Scale Template Matching Fallback ─────────────────
        tmpl_bbox, tmpl_score = self._match_template(frame)
        if (tmpl_bbox is not None
                and tmpl_score >= self.TEMPLATE_THRESHOLD
                and self.validate_bbox(tmpl_bbox, (fh, fw))):
            # HSV Verification on candidate
            hsv_score = self._verify_hsv(frame, tmpl_bbox)
            if hsv_score >= self.HSV_CORR_THRESHOLD:
                # Combined confidence for Template
                confidence = 0.50 * tmpl_score + 0.50 * hsv_score
                if confidence >= self.RECOVERY_THRESHOLD:
                    return True, tmpl_bbox, float(confidence), "Template"

        # Weak evidence / No match found -> Keep searching
        return False, None, 0.0, "None"

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: ORB Matching with Lowe Ratio & RANSAC Homography
    # ─────────────────────────────────────────────────────────────────────────
    def _match_orb(self, frame):
        """
        ORB matching pipeline:
          1. Detect ORB features in frame.
          2. Match with BFMatcher + Lowe's Ratio Test.
          3. Geometric check: RANSAC Homography + polygon convexity.
          4. Centroid fallback if Homography is degenerate.
        """
        if self.ref_descriptors is None or len(self.ref_descriptors) < 4:
            return None, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.orb.detectAndCompute(gray, None)

        if frame_des is None or len(frame_des) < 2:
            return None, 0.0

        # KNN Match (k=2)
        raw_matches = self.bf.knnMatch(self.ref_descriptors, frame_des, k=2)

        # Lowe's Ratio Test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.RATIO_THRESHOLD * n.distance:
                    good.append(m)

        n_good = len(good)
        if n_good < self.MIN_ORB_MATCHES:
            return None, 0.0

        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Geometric Check via RANSAC Homography
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None and mask is not None:
                n_inliers = int(mask.sum())
                inlier_ratio = n_inliers / n_good

                if inlier_ratio >= self.MIN_INLIER_RATIO:
                    corners = np.float32([
                        [0, 0],
                        [self.ref_w, 0],
                        [self.ref_w, self.ref_h],
                        [0, self.ref_h],
                    ]).reshape(-1, 1, 2)

                    dst_corners = cv2.perspectiveTransform(corners, H)

                    # Verify projected quad is convex (geometric validity)
                    if cv2.isContourConvex(np.int32(dst_corners)):
                        xs = dst_corners[:, 0, 0]
                        ys = dst_corners[:, 0, 1]
                        x_min, x_max = float(xs.min()), float(xs.max())
                        y_min, y_max = float(ys.min()), float(ys.max())

                        bbox = (int(x_min), int(y_min),
                                int(x_max - x_min), int(y_max - y_min))
                        score = min(1.0, n_good / 30.0) * inlier_ratio
                        return bbox, float(score)
        except cv2.error:
            pass

        # Centroid Fallback (when Homography is degenerate but matches are clustered)
        coords = dst_pts.reshape(-1, 2)
        cx = float(np.median(coords[:, 0]))
        cy = float(np.median(coords[:, 1]))
        bbox = (int(cx - self.ref_w / 2), int(cy - self.ref_h / 2),
                self.ref_w, self.ref_h)
        score = min(1.0, n_good / 30.0) * 0.45
        return bbox, float(score)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Multi-Scale Template Matching
    # ─────────────────────────────────────────────────────────────────────────
    def _match_template(self, frame):
        """
        Multi-scale template matching across template history.
        Uses cv2.TM_CCOEFF_NORMED with standard deviation guard.
        """
        if not self.template_history:
            return None, 0.0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]

        best_score = -1.0
        best_bbox  = None

        for tmpl_bgr in self.template_history:
            gray_tmpl = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
            # Guard against degenerate uniform templates
            if gray_tmpl.std() < 2.0:
                continue

            th, tw = gray_tmpl.shape

            for scale in self.TEMPLATE_SCALES:
                nw = max(10, int(tw * scale))
                nh = max(10, int(th * scale))

                if nw >= fw or nh >= fh:
                    continue

                scaled = cv2.resize(gray_tmpl, (nw, nh))

                try:
                    result = cv2.matchTemplate(gray_frame, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        best_bbox = (max_loc[0], max_loc[1], nw, nh)
                except cv2.error:
                    continue

        normalised_score = max(0.0, float(best_score))
        return best_bbox, normalised_score

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Candidate Bounding Box Validation
    # ─────────────────────────────────────────────────────────────────────────
    def validate_bbox(self, bbox, frame_shape):
        """
        Validates candidate bounding box dimensions, screen limits,
        and scale ratio relative to original reference target.
        """
        if bbox is None:
            return False

        x, y, w, h = bbox
        fh, fw = frame_shape[:2]

        # Minimum dimensions
        if w < 10 or h < 10:
            return False

        # Must have overlap with frame boundary
        if x < -w or y < -h or x > fw or y > fh:
            return False

        # Scale ratio check vs. reference area
        ref_area = max(1.0, float(self.ref_w * self.ref_h))
        cand_area = float(w * h)
        scale_ratio = cand_area / ref_area

        if not (0.15 <= scale_ratio <= 5.0):
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: HSV Color Verification
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_hsv_hist(bgr_roi):
        """
        2D Hue-Saturation histogram (30 x 32 bins), normalized.
        Value channel excluded to resist lighting changes.
        """
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _verify_hsv(self, frame, bbox):
        """
        Compares candidate patch HSV histogram to reference using Pearson correlation.
        Returns float in [0.0, 1.0].
        """
        if self.ref_hist is None:
            return 1.0

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        cand_hist = self._compute_hsv_hist(roi)
        corr = cv2.compareHist(self.ref_hist, cand_hist, cv2.HISTCMP_CORREL)
        return max(0.0, float(corr))

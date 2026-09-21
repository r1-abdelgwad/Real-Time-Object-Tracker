"""
Re-Identification (Re-ID) Module
=================================
Author : Rawan Abdelgwad
Date   : 2026

Architecture — Two-Path Recovery with HSV Gating
-------------------------------------------------
Path A  (primary)  : ORB + BFMatcher + Lowe Ratio Test + RANSAC Homography
Path B  (fallback) : Multi-scale Template Matching against template history

Both paths are gated by an HSV colour-histogram check before accepting
a candidate.  This prevents recovering onto an object that has similar
texture/edges but a different colour.

Combined confidence score
    Path A :  0.65 * orb_score  + 0.35 * hsv_score
    Path B :  0.50 * tmpl_score + 0.50 * hsv_score
A recovery is only accepted when combined_confidence >= RECOVERY_THRESHOLD.

Appearance drift protection
    Templates are only added to history when update_profile() is called
    with confidence >= PROFILE_UPDATE_MIN_CONF.  ORB reference descriptors
    and the HSV reference histogram are NEVER overwritten after the initial
    selection — this anchors identity regardless of lighting changes.
"""

import cv2
import numpy as np


class ReIDMatcher:
    # ── Tunable thresholds ─────────────────────────────────────────────────
    ORB_FEATURES            = 500    # max ORB keypoints in reference ROI
    MIN_ORB_MATCHES         = 8      # good matches needed to attempt homography
    RATIO_THRESHOLD         = 0.72   # Lowe ratio test (lower = stricter)
    MIN_INLIER_RATIO        = 0.30   # RANSAC inliers / good_matches floor
    HSV_CORR_THRESHOLD      = 0.50   # min HSV correlation to accept candidate
    TEMPLATE_THRESHOLD      = 0.50   # min TM_CCOEFF_NORMED score to attempt HSV check
    RECOVERY_THRESHOLD      = 0.52   # min combined confidence to confirm recovery
    MAX_TEMPLATES           = 5      # template history depth (FIFO)
    PROFILE_UPDATE_MIN_CONF = 0.75   # min confidence required to add a new template
    TEMPLATE_SCALES         = [0.75, 0.875, 1.0, 1.125, 1.25]  # multi-scale search

    # ── Constructor ────────────────────────────────────────────────────────
    def __init__(self):
        # ORB detector + Brute-Force matcher (Hamming distance for binary descriptors)
        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

        # ORB reference (set once at selection, never overwritten)
        self.ref_keypoints   = None   # list[KeyPoint]
        self.ref_descriptors = None   # ndarray shape (N, 32), dtype uint8
        self.ref_w           = 0
        self.ref_h           = 0

        # HSV reference histogram (set once at selection, never overwritten)
        self.ref_hist = None

        # Template history: list of BGR patches, newest first
        # Starts with the initial ROI; updated by update_profile()
        self.template_history = []

    # ── Initial feature extraction ─────────────────────────────────────────
    def extract_reference_features(self, frame, bbox):
        """
        Called once when the user selects the target.
        Extracts ORB descriptors, HSV histogram, and initial template — all
        stored as immutable identity anchors (never overwritten after this).
        """
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x),     max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # --- ORB reference (primary Re-ID signal) ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray, None)
        self.ref_w = x2 - x1
        self.ref_h = y2 - y1

        # --- HSV histogram reference (colour gate) ---
        self.ref_hist = self._compute_hsv_hist(roi)

        # --- Initial template (Path B seed) ---
        self.template_history = [roi.copy()]

    # ── Profile update (appearance drift protection) ───────────────────────
    def update_profile(self, frame, bbox, confidence):
        """
        Append a new template patch to history only when tracking is stable.
        Called by tracker.py on a fixed interval with a confidence value.

        Rules:
          - Confidence must be >= PROFILE_UPDATE_MIN_CONF.
          - ORB descriptors and HSV histogram are NEVER updated here.
            Updating them risks drifting the identity anchor.
          - Only templates drift (slightly) to adapt to slow appearance changes.
        """
        if confidence < self.PROFILE_UPDATE_MIN_CONF:
            return

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x),     max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return

        self.template_history.insert(0, patch.copy())        # newest first
        if len(self.template_history) > self.MAX_TEMPLATES:
            self.template_history.pop()                       # drop oldest

    # ── HSV histogram helpers ──────────────────────────────────────────────
    @staticmethod
    def _compute_hsv_hist(bgr_roi):
        """
        2-D Hue-Saturation histogram (30 x 32 bins), L2-normalised.
        Value channel excluded — makes the histogram robust to lighting changes.
        """
        hsv  = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def _verify_hsv(self, frame, bbox):
        """
        Compute HSV histogram of the candidate region and compare it against
        the reference using Pearson correlation.

        Returns float in [0, 1].  Returns 1.0 if no reference exists (bypass).
        """
        if self.ref_hist is None:
            return 1.0

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x),     max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        cand_hist   = self._compute_hsv_hist(roi)
        correlation = cv2.compareHist(self.ref_hist, cand_hist, cv2.HISTCMP_CORREL)
        return max(0.0, float(correlation))

    # ── Path A: ORB matching ───────────────────────────────────────────────
    def _match_orb(self, frame):
        """
        Full ORB pipeline:
          1. Detect + compute descriptors in current frame.
          2. knn-match against stored reference descriptors.
          3. Filter with Lowe ratio test.
          4. Estimate location via RANSAC Homography.
          5. Fallback to keypoint centroid if Homography degenerates.

        Returns:
          bbox  (tuple|None)  : (x, y, w, h) candidate
          score (float)       : normalised score in [0, 1]
          n_good (int)        : number of good matches (for logging)
        """
        if self.ref_descriptors is None or len(self.ref_descriptors) < 4:
            return None, 0.0, 0

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.orb.detectAndCompute(gray, None)

        if frame_des is None or len(frame_des) < 2:
            return None, 0.0, 0

        # knn match + Lowe ratio test
        raw_matches = self.bf.knnMatch(self.ref_descriptors, frame_des, k=2)
        good = []
        for tup in raw_matches:
            if len(tup) == 2:
                m, n = tup
                if m.distance < self.RATIO_THRESHOLD * n.distance:
                    good.append(m)

        n_good = len(good)
        if n_good < self.MIN_ORB_MATCHES:
            return None, 0.0, n_good

        src_pts = np.float32(
            [self.ref_keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [frame_kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        # RANSAC Homography
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None and mask is not None:
                n_inliers    = int(mask.sum())
                inlier_ratio = n_inliers / n_good

                if inlier_ratio >= self.MIN_INLIER_RATIO:
                    corners = np.float32([
                        [0,          0         ],
                        [self.ref_w, 0         ],
                        [self.ref_w, self.ref_h],
                        [0,          self.ref_h],
                    ]).reshape(-1, 1, 2)
                    dst_corners = cv2.perspectiveTransform(corners, H)
                    xs, ys = dst_corners[:, 0, 0], dst_corners[:, 0, 1]

                    x_min, x_max = float(xs.min()), float(xs.max())
                    y_min, y_max = float(ys.min()), float(ys.max())
                    bbox  = (int(x_min), int(y_min),
                             int(x_max - x_min), int(y_max - y_min))
                    # score: normalised match count * inlier quality
                    score = min(1.0, n_good / 30.0) * inlier_ratio
                    return bbox, score, n_good
        except cv2.error:
            pass

        # Centroid fallback (homography failed / degenerate)
        coords  = dst_pts.reshape(-1, 2)
        cx, cy  = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))
        bbox    = (int(cx - self.ref_w / 2), int(cy - self.ref_h / 2),
                   self.ref_w, self.ref_h)
        score   = min(1.0, n_good / 30.0) * 0.45    # discounted: no geometric check
        return bbox, score, n_good

    # ── Path B: Template matching ──────────────────────────────────────────
    def _match_template(self, frame):
        """
        Multi-scale template matching over the full template history.
        Tries TEMPLATE_SCALES of each stored template.
        Uses TM_CCOEFF_NORMED (range -1 to 1; higher = better).

        Returns:
          bbox  (tuple|None) : (x, y, w, h) best candidate
          score (float)      : best normalised match score in [0, 1]
        """
        if not self.template_history:
            return None, 0.0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw     = frame.shape[:2]

        best_score = -1.0
        best_bbox  = None

        for tmpl_bgr in self.template_history:
            gray_tmpl  = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
            if gray_tmpl.std() < 2.0:
                continue
            th, tw     = gray_tmpl.shape

            for scale in self.TEMPLATE_SCALES:
                nw = max(10, int(tw * scale))
                nh = max(10, int(th * scale))

                # Template must be smaller than frame
                if nw >= fw or nh >= fh:
                    continue

                scaled = cv2.resize(gray_tmpl, (nw, nh))

                try:
                    result  = cv2.matchTemplate(gray_frame, scaled,
                                                cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        best_bbox  = (max_loc[0], max_loc[1], nw, nh)
                except cv2.error:
                    continue

        # Normalise from [-1, 1] to [0, 1]
        normalised_score = max(0.0, float(best_score))
        return best_bbox, normalised_score

    # ── Bbox validation ────────────────────────────────────────────────────
    def validate_bbox(self, bbox, frame_shape):
        """
        Reject geometrically implausible candidates.
        Checks: minimum size, within-frame bounds, scale ratio vs reference.
        """
        if bbox is None:
            return False

        x, y, w, h = bbox
        fh, fw     = frame_shape[:2]

        if w < 10 or h < 10:
            return False

        # Allow partial off-screen presence (object near edge)
        if x < -w or y < -h or x > fw or y > fh:
            return False

        ref_area   = max(1.0, float(self.ref_w * self.ref_h))
        cand_area  = float(w * h)
        scale      = cand_area / ref_area

        # Reject if scale is wildly different (0.15x to 5x of reference area)
        if not (0.15 <= scale <= 5.0):
            return False

        return True

    # ── Main recovery entry point ──────────────────────────────────────────
    def recover_object(self, frame):
        """
        Attempt to relocate the lost target in the given frame.

        Path A (ORB) runs first — it is more precise.
        Path B (Template) runs as fallback when ORB has too few features
        (e.g. featureless/uniform-colour objects).
        Both paths are gated by HSV colour verification.

        Returns:
          recovered   (bool)        : True if a valid candidate was found
          bbox        (tuple|None)  : (x, y, w, h) of recovered location
          confidence  (float)       : combined score in [0, 1]
          method      (str)         : 'ORB' | 'Template' | 'None'
        """
        # ── Path A: ORB ────────────────────────────────────────────────────
        orb_bbox, orb_score, n_good = self._match_orb(frame)

        if orb_bbox is not None and self.validate_bbox(orb_bbox, frame.shape):
            hsv_score  = self._verify_hsv(frame, orb_bbox)
            if hsv_score >= self.HSV_CORR_THRESHOLD:
                confidence = 0.65 * orb_score + 0.35 * hsv_score
                if confidence >= self.RECOVERY_THRESHOLD:
                    return True, orb_bbox, confidence, "ORB"

        # ── Path B: Template matching (fallback) ───────────────────────────
        tmpl_bbox, tmpl_score = self._match_template(frame)

        if (tmpl_bbox is not None
                and tmpl_score >= self.TEMPLATE_THRESHOLD
                and self.validate_bbox(tmpl_bbox, frame.shape)):
            hsv_score  = self._verify_hsv(frame, tmpl_bbox)
            if hsv_score >= self.HSV_CORR_THRESHOLD:
                confidence = 0.50 * tmpl_score + 0.50 * hsv_score
                if confidence >= self.RECOVERY_THRESHOLD:
                    return True, tmpl_bbox, confidence, "Template"

        return False, None, 0.0, "None"

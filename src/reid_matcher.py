"""
Re-Identification (Re-ID) & Feature Matching Module
===================================================
Author  : Rawan Abdelgwad
Date    : 2026

Handles ORB keypoint extraction, descriptor matching (BFMatcher + Ratio Test),
Homography transformation via RANSAC, and bounding box validation.
"""

import cv2
import numpy as np


class ReIDMatcher:
    def __init__(self, max_features=500, min_good_matches=6, ratio_threshold=0.75):
        self.max_features = max_features
        self.min_good_matches = min_good_matches
        self.ratio_threshold = ratio_threshold
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.ref_keypoints = []
        self.ref_descriptors = None
        self.ref_w = 0
        self.ref_h = 0

    def extract_reference_features(self, frame, bbox):
        """Extract reference ORB descriptors from initial user-selected ROI."""
        x, y, w, h = [int(v) for v in bbox]
        frame_h, frame_w = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            self.ref_keypoints, self.ref_descriptors = [], None
            self.ref_w, self.ref_h = w, h
            return

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(gray_roi, None)
        self.ref_w = w
        self.ref_h = h

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

        ref_area = max(1.0, float(self.ref_w * self.ref_h))
        cand_area = float(w * h)
        scale_ratio = cand_area / ref_area

        if scale_ratio < 0.2 or scale_ratio > 4.5:
            return False

        return True

    def recover_object(self, frame):
        """Match current frame features against stored reference features."""
        if self.ref_descriptors is None or len(self.ref_descriptors) < 4:
            return False, None, 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.orb.detectAndCompute(gray_frame, None)

        if frame_des is None or len(frame_des) < 2:
            return False, None, 0

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

        # Method 1: Homography RANSAC
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                pts = np.float32([[0, 0], [self.ref_w, 0], [self.ref_w, self.ref_h], [0, self.ref_h]]).reshape(-1, 1, 2)
                dst_corners = cv2.perspectiveTransform(pts, H)

                x_min = float(np.min(dst_corners[:, 0, 0]))
                y_min = float(np.min(dst_corners[:, 0, 1]))
                x_max = float(np.max(dst_corners[:, 0, 0]))
                y_max = float(np.max(dst_corners[:, 0, 1]))

                cand_w = x_max - x_min
                cand_h = y_max - y_min
                cand_bbox = (int(x_min), int(y_min), int(cand_w), int(cand_h))

                if self.validate_bbox(cand_bbox, frame.shape):
                    return True, cand_bbox, match_count
        except cv2.error:
            pass

        # Method 2: Centroid estimation
        dst_coords = dst_pts.reshape(-1, 2)
        center_x = float(np.median(dst_coords[:, 0]))
        center_y = float(np.median(dst_coords[:, 1]))

        cand_x = int(center_x - self.ref_w / 2.0)
        cand_y = int(center_y - self.ref_h / 2.0)
        cand_bbox = (cand_x, cand_y, int(self.ref_w), int(self.ref_h))

        if self.validate_bbox(cand_bbox, frame.shape):
            return True, cand_bbox, match_count

        return False, None, match_count

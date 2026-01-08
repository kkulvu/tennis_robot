"""
Optimized and Stable SORT for Tennis Tracking
- Ensures unique ID stability
- Handles empty detections gracefully
- Suitable for hitting detection
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """Compute IOU between two boxes [x1,y1,x2,y2]"""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    return inter / (area1 + area2 - inter + 1e-6)

class KalmanBoxTracker:
    """Represents a tracked tennis ball"""
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        # Measurement function
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.

        x1, y1, x2, y2 = bbox[:4]
        x = (x1 + x2) / 2.
        y = (y1 + y2) / 2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        self.kf.x[:4] = np.array([x, y, s, r]).reshape((4,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        x1, y1, x2, y2 = bbox[:4]
        x = (x1 + x2)/2.
        y = (y1 + y2)/2.
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1)/(y2 - y1 + 1e-6)
        z = np.array([x, y, s, r]).reshape((4,1))
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return self.get_state()

    def get_state(self):
        x, y, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        return np.array([x - w/2., y - h/2., x + w/2., y + h/2.])

class Sort:
    def __init__(self, max_age=15, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0,5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = pos
            trks[t][4] = 0
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # 更新匹配到的跟踪器
        for t, trk in enumerate(self.trackers):
            if matched.size > 0 and t in matched[:,1]:
                d = matched[matched[:,1]==t,0][0]
                trk.update(dets[d])

        # 创建新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        ret = []
        for trk in self.trackers:
            d = trk.get_state()
            if trk.time_since_update <= self.max_age:
                ret.append(np.concatenate((d, [trk.id])).reshape(1,-1))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(dets)), np.empty((0), dtype=int)

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d,t] = iou(det, trk)

        if iou_matrix.size > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0,2), dtype=int)

        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trks)))
        matches = []

        if matched_indices.size > 0:
            for d, t in matched_indices:
                if iou_matrix[d,t] < self.iou_threshold:
                    continue
                matches.append([d,t])
                if d in unmatched_dets: unmatched_dets.remove(d)
                if t in unmatched_trks: unmatched_trks.remove(t)

        matches = np.array(matches) if matches else np.empty((0,2), dtype=int)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

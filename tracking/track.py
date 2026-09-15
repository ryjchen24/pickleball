from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

DEFAULT_ARGS = dict(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


@dataclass
class Detections:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    @classmethod
    def from_arrays(cls, xyxy, conf, classes=None):
        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        classes = np.zeros(len(conf), dtype=np.float32) if classes is None else np.asarray(classes, dtype=np.float32)
        return cls(xyxy, conf, classes.reshape(-1))

    @classmethod
    def from_ultralytics(cls, boxes):
        return cls.from_arrays(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())

    @property
    def xywh(self):
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([xy, wh], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask])

    def filter_classes(self, classes):
        if classes is None:
            return self
        return self[np.isin(self.cls, list(classes))]


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    score: float
    cls: int


class Tracker:
    def __init__(self, classes=None, **overrides):
        self.classes = classes
        self.args = SimpleNamespace(**{**DEFAULT_ARGS, **overrides})
        self.reset()

    def reset(self):
        self.tracker = BYTETracker(self.args)

    def update(self, detections):
        detections = detections.filter_classes(self.classes)
        out = self.tracker.update(detections)
        return [
            Track(int(row[4]), row[:4].copy(), float(row[5]), int(row[6]))
            for row in np.asarray(out).reshape(-1, 8)
        ]


def track_sequence(frames, classes=None, **overrides):
    tracker = Tracker(classes, **overrides)
    history = defaultdict(list)
    for frame_idx, detections in enumerate(frames):
        for t in tracker.update(detections):
            history[t.track_id].append((frame_idx, t.xyxy))
    return dict(history)

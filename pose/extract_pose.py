import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmlib import RTMPose
from ultralytics import YOLO

from shot_classifier.graph_utils import COCO_EDGES, NUM_JOINTS
from tracking.track import Detections, track_sequence

MODELS = {
    "s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    "m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
}
INPUT_SIZE = (192, 256)
PERSON_CLASS = 0


class PoseEstimator:
    def __init__(self, model="m", device="cpu"):
        self.model = RTMPose(MODELS.get(model, model), model_input_size=INPUT_SIZE, device=device)

    def __call__(self, image, boxes):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
        keypoints, scores = self.model(image, bboxes=boxes.tolist())
        return np.concatenate([keypoints, scores[..., None]], axis=-1).astype(np.float32)


class PersonDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.4):
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, image):
        result = self.model.predict(image, classes=[PERSON_CLASS], conf=self.conf, verbose=False)[0]
        return Detections.from_ultralytics(result.boxes)


def extract_track_poses(video_path, history, estimator):
    by_frame = {}
    for track_id, entries in history.items():
        for frame_idx, xyxy in entries:
            by_frame.setdefault(frame_idx, []).append((track_id, xyxy))

    spans = {tid: (entries[0][0], entries[-1][0]) for tid, entries in history.items()}
    poses = {
        tid: np.full((end - start + 1, NUM_JOINTS, 3), np.nan, dtype=np.float32)
        for tid, (start, end) in spans.items()
    }

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in by_frame:
            ids, boxes = zip(*by_frame[frame_idx])
            for tid, kp in zip(ids, estimator(frame, np.stack(boxes))):
                poses[tid][frame_idx - spans[tid][0]] = kp
        frame_idx += 1
    cap.release()
    return {tid: (spans[tid][0], kp) for tid, kp in poses.items()}


def save_track_poses(poses, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for tid, (start, keypoints) in poses.items():
        path = os.path.join(out_dir, f"{prefix}_track{tid:03d}_f{start:05d}.npy")
        np.save(path, keypoints)
        paths.append(path)
    return paths


def draw_skeleton(image, keypoints, min_score=0.3, color=(0, 255, 0)):
    out = image.copy()
    for person in keypoints:
        for i, j in COCO_EDGES:
            if person[i, 2] >= min_score and person[j, 2] >= min_score:
                p1 = tuple(int(v) for v in person[i, :2])
                p2 = tuple(int(v) for v in person[j, :2])
                cv2.line(out, p1, p2, color, 2)
        for x, y, s in person:
            if s >= min_score:
                cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    return out


def run_image(path, out_dir, detector, estimator):
    image = cv2.imread(path)
    detections = detector(image)
    keypoints = estimator(image, detections.xyxy)
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}_pose.npy"), keypoints)
    cv2.imwrite(os.path.join(out_dir, f"{name}_pose.jpg"), draw_skeleton(image, keypoints))
    return keypoints


def run_video(path, out_dir, detector, estimator):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(detector(frame))
    cap.release()
    history = track_sequence(frames)
    poses = extract_track_poses(path, history, estimator)
    name = os.path.splitext(os.path.basename(path))[0]
    return save_track_poses(poses, out_dir, name)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--video")
    parser.add_argument("--out-dir", default="data/processed/pose_sequences")
    parser.add_argument("--model", default="m")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    detector = PersonDetector(args.detector)
    estimator = PoseEstimator(args.model, args.device)
    if args.image:
        keypoints = run_image(args.image, args.out_dir, detector, estimator)
        print(f"{len(keypoints)} people -> {args.out_dir}")
    else:
        paths = run_video(args.video, args.out_dir, detector, estimator)
        print(f"{len(paths)} tracks -> {args.out_dir}")


if __name__ == "__main__":
    main()

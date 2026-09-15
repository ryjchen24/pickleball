import numpy as np

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_JOINTS = len(COCO_KEYPOINTS)

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def edge_matrix(num_joints=NUM_JOINTS, edges=COCO_EDGES):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A


def normalize(A):
    degree = A.sum(axis=0)
    degree[degree == 0] = 1.0
    return A / degree


def build_adjacency(num_joints=NUM_JOINTS, edges=COCO_EDGES):
    neighbors = normalize(edge_matrix(num_joints, edges))
    return np.stack([np.eye(num_joints, dtype=np.float32), neighbors])

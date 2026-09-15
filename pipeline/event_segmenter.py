from dataclasses import dataclass

import numpy as np


@dataclass
class Contact:
    frame: int
    score: float
    start: int
    end: int


def fill_gaps(traj, max_gap=5):
    traj = np.asarray(traj, dtype=np.float64).copy()
    missing = np.isnan(traj).any(axis=1)
    if missing.all():
        return traj
    idx = np.arange(len(traj))
    known = idx[~missing]
    i = 0
    while i < len(traj):
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < len(traj) and missing[j]:
            j += 1
        if i > 0 and j < len(traj) and j - i <= max_gap:
            for d in range(traj.shape[1]):
                traj[i:j, d] = np.interp(idx[i:j], known, traj[~missing, d])
        i = j
    return traj


def smooth(traj, window=3):
    if window <= 1:
        return traj
    kernel = np.ones(window)
    out = np.empty_like(traj)
    for d in range(traj.shape[1]):
        col = traj[:, d]
        valid = ~np.isnan(col)
        num = np.convolve(np.where(valid, col, 0.0), kernel, mode="same")
        den = np.convolve(valid.astype(float), kernel, mode="same")
        out[:, d] = np.where(valid & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return out


def turn_scores(traj, lag=3, min_speed=2.0):
    n = len(traj)
    angles = np.zeros(n)
    speed_ratio = np.ones(n)
    for t in range(lag, n - lag):
        before = traj[t] - traj[t - lag]
        after = traj[t + lag] - traj[t]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        s0, s1 = np.linalg.norm(before) / lag, np.linalg.norm(after) / lag
        if max(s0, s1) < min_speed:
            continue
        if s0 > 1e-9 and s1 > 1e-9:
            cos = np.clip(before @ after / (np.linalg.norm(before) * np.linalg.norm(after)), -1.0, 1.0)
            angles[t] = np.degrees(np.arccos(cos))
        speed_ratio[t] = max(s0, s1) / max(min(s0, s1), 1e-9)
    return angles, speed_ratio


def non_max_suppression(candidates, scores, min_separation):
    kept = []
    for t in sorted(candidates, key=lambda t: -scores[t]):
        if all(abs(t - k) >= min_separation for k in kept):
            kept.append(t)
    return sorted(kept)


def window_bounds(frame, n_frames, window):
    half = window // 2
    start = max(0, frame - half)
    end = min(n_frames, start + window)
    start = max(0, end - window)
    return start, end


def detect_contacts(
    traj,
    window=24,
    lag=3,
    angle_thresh=45.0,
    speed_ratio_thresh=2.5,
    min_speed=2.0,
    min_separation=15,
    max_gap=5,
    smooth_window=3,
):
    traj = smooth(fill_gaps(traj, max_gap), smooth_window)
    angles, ratios = turn_scores(traj, lag, min_speed)
    score = np.maximum(angles / angle_thresh, ratios / speed_ratio_thresh)
    candidates = [t for t in range(len(traj)) if score[t] >= 1.0]
    peaks = [
        t for t in candidates
        if score[t] >= score[max(0, t - 1)] and score[t] >= score[min(len(traj) - 1, t + 1)]
    ]
    kept = non_max_suppression(peaks, score, min_separation)
    return [Contact(t, float(score[t]), *window_bounds(t, len(traj), window)) for t in kept]

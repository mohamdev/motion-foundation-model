#!/usr/bin/env python3
"""
Visualize continuous motion by reconstructing from sliding window sequences.

This script takes 60-frame sliding window sequences (with 30-frame stride)
and reconstructs the full continuous motion for visualization.

Usage
-----
python visualize_continuous_motion.py \
  --dataset-root ./datasets/dataset0 \
  --subject 3 \
  --fps 30

Notes
-----
- Sequences are assumed to use 30-frame sliding window stride (50% overlap)
- The continuous motion is reconstructed by stitching overlapping windows
- Motions are centered on midHip for better visualization
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# ===============================================================
# Joint configuration
# ===============================================================

JOINT_NAMES = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "midHip",
    "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
    "RSmallToe", "LSmallToe", "RBigToe", "LBigToe",
    "RElbow", "LElbow", "RWrist", "LWrist",
]

Neck, RSh, LSh, RHip, LHip, midHip = 0, 1, 2, 3, 4, 5
RKnee, LKnee, RAnk, LAnk, RHeel, LHeel = 6, 7, 8, 9, 10, 11
RST, LST, RBT, LBT, RElb, LElb, RWri, LWri = 12, 13, 14, 15, 16, 17, 18, 19

EDGES = [
    (midHip, Neck), (midHip, RHip), (midHip, LHip), (Neck, RSh), (Neck, LSh),
    (RSh, RElb), (RElb, RWri),
    (LSh, LElb), (LElb, LWri),
    (RHip, RKnee), (RKnee, RAnk), (RAnk, RHeel), (RAnk, RST), (RAnk, RBT),
    (RHeel, RST), (RHeel, RBT),
    (LHip, LKnee), (LKnee, LAnk), (LAnk, LHeel), (LAnk, LST), (LAnk, LBT),
    (LHeel, LST), (LHeel, LBT),
]


# ===============================================================
# Data loading
# ===============================================================

def load_subjects(info_path: Path) -> np.ndarray:
    info = np.load(info_path, allow_pickle=True)
    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()
    return np.asarray(info["subjects"])


def load_features(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["data/features"][:]


def index_by_subject(subjects: np.ndarray) -> Dict[int, np.ndarray]:
    d = {}
    for i, s in enumerate(subjects):
        d.setdefault(int(s), []).append(i)
    for k in d:
        d[k] = np.asarray(d[k], dtype=int)
    return d


# ===============================================================
# Layout detection and preprocessing
# ===============================================================

def strip_height_weight(features: np.ndarray) -> np.ndarray:
    N, T, F = features.shape
    if F % 3 == 2:
        return features[:, :, : F - 2]
    return features


def to_xyz_interleaved(cols: np.ndarray) -> np.ndarray:
    N, T, W = cols.shape
    K = W // 3
    return cols.reshape(N, T, K, 3)


def to_xyz_stacked(cols: np.ndarray) -> np.ndarray:
    N, T, W = cols.shape
    K = W // 3
    xs = cols[:, :, :K]
    ys = cols[:, :, K:2 * K]
    zs = cols[:, :, 2 * K:3 * K]
    return np.stack([xs, ys, zs], axis=-1)


def bone_length_score(xyz: np.ndarray, edges: List[Tuple[int, int]]) -> float:
    T, K, _ = xyz.shape
    scores = []
    for i, j in edges:
        if i < K and j < K:
            dij = np.linalg.norm(xyz[:, i, :] - xyz[:, j, :], axis=1)
            m = np.mean(dij) + 1e-8
            s = np.std(dij)
            scores.append(s / m)
    return float(np.mean(scores)) if scores else float("inf")


def features_to_xyz_autolayout(features: np.ndarray, force_layout: Optional[str] = None):
    cols = strip_height_weight(features)
    N, T, W = cols.shape
    if W % 3 != 0:
        usable = W - (W % 3)
        cols = cols[:, :, :usable]
        W = usable
    K = W // 3

    if force_layout in ("interleaved", "stacked"):
        layout = force_layout
    else:
        xyz_i = to_xyz_interleaved(cols)
        xyz_s = to_xyz_stacked(cols)
        si = bone_length_score(xyz_i[0], EDGES)
        ss = bone_length_score(xyz_s[0], EDGES)
        layout = "interleaved" if si <= ss else "stacked"
        print(f"[info] Auto-selected layout: {layout} (scores: interleaved={si:.4f}, stacked={ss:.4f})")

    xyz = to_xyz_interleaved(cols) if layout == "interleaved" else to_xyz_stacked(cols)
    return xyz, layout


# ===============================================================
# Continuous motion reconstruction
# ===============================================================

def detect_sliding_window_stride(sequences: np.ndarray) -> int:
    """
    Detect the stride of sliding window by comparing consecutive sequences.

    Args:
        sequences: (N, T, K, 3) array of sequences

    Returns:
        stride: Number of frames between consecutive windows
    """
    if len(sequences) < 2:
        return sequences.shape[1]  # Single sequence, return full length

    seq1 = sequences[0]
    seq2 = sequences[1]
    T = seq1.shape[0]

    # Check for overlap by comparing seq2[0] with seq1[offset]
    for offset in range(1, T):
        if np.allclose(seq1[offset], seq2[0], atol=1e-6):
            print(f"[info] Detected sliding window stride: {offset} frames")
            return offset

    # No overlap found - assume non-overlapping sequences
    print(f"[info] No overlap detected - using stride of {T} frames")
    return T


def reconstruct_continuous_motion(sequences: np.ndarray, stride: Optional[int] = None) -> np.ndarray:
    """
    Reconstruct continuous motion from sliding window sequences.

    Args:
        sequences: (N, T, K, 3) array of N sequences, each T frames long
        stride: Sliding window stride (auto-detected if None)

    Returns:
        continuous_motion: (T_total, K, 3) array of continuous motion
    """
    N, T, K, _ = sequences.shape

    if N == 1:
        # Single sequence, return as is
        return sequences[0]

    # Auto-detect stride if not provided
    if stride is None:
        stride = detect_sliding_window_stride(sequences)

    # Calculate total length
    total_length = T + (N - 1) * stride

    # Initialize output array
    continuous = np.zeros((total_length, K, 3), dtype=sequences.dtype)
    weight = np.zeros((total_length, K, 3), dtype=np.float32)

    # Accumulate overlapping sequences with averaging
    for i, seq in enumerate(sequences):
        start_idx = i * stride
        end_idx = start_idx + T
        continuous[start_idx:end_idx] += seq
        weight[start_idx:end_idx] += 1.0

    # Average overlapping regions
    mask = weight > 0
    continuous[mask] /= weight[mask]

    print(f"[info] Reconstructed continuous motion: {N} sequences × {T} frames → {total_length} frames")
    return continuous


# ===============================================================
# Visualization
# ===============================================================

def setup_scene(vis, radius: float, K: int):
    sphere = g.Sphere(radius)
    joint_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True)
    for j in range(K):
        vis[f"joints/{j}"].set_object(sphere, joint_mat)
    try:
        vis["/Grid"].set_object(g.GridHelper(size=2.0, divisions=20))
    except Exception:
        pass


def play_continuous_motion(vis, motion: np.ndarray, fps: int, edges: List[Tuple[int, int]],
                          reference_joint: int = midHip):
    """
    Play continuous motion in realtime.

    Args:
        vis: MeshCat visualizer
        motion: (T, K, 3) continuous motion array
        fps: Frames per second
        edges: List of (i, j) tuples defining skeleton edges
        reference_joint: Joint index to center motion on
    """
    T, K, _ = motion.shape
    dt = 1.0 / float(fps)
    line_mat = g.LineBasicMaterial(linewidth=2)

    # Center on first frame's reference joint
    if 0 <= reference_joint < K:
        origin = motion[0, reference_joint:reference_joint+1, :]
    else:
        origin = motion[0].mean(axis=0, keepdims=True)

    motion_centered = motion - origin

    print(f"[info] Playing continuous motion: {T} frames at {fps} fps ({T/fps:.1f} seconds)")

    for t in range(T):
        # Update joint positions
        for j in range(K):
            vis[f"joints/{j}"].set_transform(
                tf.translation_matrix(motion_centered[t, j].tolist())
            )

        # Update edges
        for e_idx, (i, j) in enumerate(edges):
            if i < K and j < K:
                p0 = motion_centered[t, i]
                p1 = motion_centered[t, j]
                pts = np.column_stack([p0, p1]).T  # (2, 3)
                geom = g.PointsGeometry(pts.T)      # (3, 2)
                vis[f"edges/{e_idx}"].set_object(g.Line(geom, line_mat))

        time.sleep(dt)


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize continuous motion from sliding window sequences")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--fps", type=int, default=300)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--layout", choices=["interleaved", "stacked"], default=None)
    parser.add_argument("--stride", type=int, default=None,
                       help="Sliding window stride (auto-detected if not specified)")
    parser.add_argument("--loop", action="store_true", help="Loop the motion continuously")
    parser.add_argument("--list", action="store_true", help="List available subjects")

    args = parser.parse_args()

    # Load data
    root = Path(args.dataset_root)
    subjects = load_subjects(root / "infoData.npy")
    features = load_features(root / "time_sequences.h5")
    by_subject = index_by_subject(subjects)

    # List subjects if requested
    if args.list:
        print("Available subjects:")
        for sid in sorted(by_subject.keys()):
            n_seq = len(by_subject[sid])
            print(f"  Subject {sid}: {n_seq} sequences")
        return

    # Check subject exists
    if args.subject not in by_subject:
        available = sorted(by_subject.keys())
        raise SystemExit(f"Subject {args.subject} not found. Available: {available}")

    # Convert to XYZ
    xyz_all, layout = features_to_xyz_autolayout(features, args.layout)
    N_total, T, K, _ = xyz_all.shape

    # Get sequences for this subject
    seq_indices = by_subject[args.subject]
    sequences = xyz_all[seq_indices]

    print(f"\n[info] Subject {args.subject}: {len(sequences)} sequences of {T} frames each, {K} joints")

    # Reconstruct continuous motion
    continuous_motion = reconstruct_continuous_motion(sequences, stride=args.stride)

    # Setup visualization
    vis = meshcat.Visualizer().open()
    setup_scene(vis, args.radius, K)
    edges = EDGES if K == 20 else [(i, i+1) for i in range(K-1)]

    print(f"[info] Layout: {layout}")
    print("[info] Press Ctrl+C to stop\n")

    # Play motion
    try:
        while True:
            play_continuous_motion(vis, continuous_motion, args.fps, edges)
            if not args.loop:
                break
            print("[info] Looping...")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[info] Stopped.")


if __name__ == "__main__":
    main()

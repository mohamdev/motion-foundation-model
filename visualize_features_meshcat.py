#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import random

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# ---------- joint set (20) ----------
NAMES = [
    "Neck","RShoulder","LShoulder","RHip","LHip","midHip",
    "RKnee","LKnee","RAnkle","LAnkle","RHeel","LHeel",
    "RSmallToe","LSmallToe","RBigToe","LBigToe",
    "RElbow","LElbow","RWrist","LWrist",
]
Neck,RSh,LSh,RHip,LHip,midHip,RKnee,LKnee,RAnk,LAnk,RHeel,LHeel,RST,LST,RBT,LBT,RElb,LElb,RWri,LWri = range(20)

EDGES = [
    # torso
    (midHip,Neck),(midHip,RHip),(midHip,LHip),(Neck,RSh),(Neck,LSh),
    # right arm
    (RSh,RElb),(RElb,RWri),
    # left arm
    (LSh,LElb),(LElb,LWri),
    # right leg + foot
    (RHip,RKnee),(RKnee,RAnk),(RAnk,RHeel),(RAnk,RST),(RAnk,RBT),(RHeel,RST),(RHeel,RBT),
    # left leg + foot
    (LHip,LKnee),(LKnee,LAnk),(LAnk,LHeel),(LAnk,LST),(LAnk,LBT),(LHeel,LST),(LHeel,LBT),
]

# ---------- data IO ----------
def load_subjects(info_path: Path) -> np.ndarray:
    info = np.load(info_path, allow_pickle=True)
    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()
    return np.asarray(info["subjects"])

def load_features(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["data/features"][:]  # (N, T, F) in your dataset

def index_by_subject(subjects: np.ndarray):
    d = {}
    for i, s in enumerate(subjects):
        d.setdefault(int(s), []).append(i)
    for k in d: d[k] = np.asarray(d[k], dtype=int)
    return d

# ---------- reshaping ----------
def strip_height_weight(features: np.ndarray):
    N,T,F = features.shape
    if F % 3 == 2:
        usable = F - 2
        print(f"[info] F={F}: stripping last 2 columns (height, weight) -> use 0:{usable}")
        return features[:,:,:usable]
    return features

def to_xyz_interleaved(cols: np.ndarray) -> np.ndarray:
    N,T,W = cols.shape
    K = W//3
    return cols.reshape(N,T,K,3)

def to_xyz_stacked(cols: np.ndarray) -> np.ndarray:
    N,T,W = cols.shape
    K = W//3
    xs = cols[:,:,:K]
    ys = cols[:,:,K:2*K]
    zs = cols[:,:,2*K:3*K]
    return np.stack([xs,ys,zs], axis=-1)

def bone_length_score(xyz: np.ndarray, edges) -> float:
    T,K,_ = xyz.shape
    scores = []
    for i,j in edges:
        dij = np.linalg.norm(xyz[:,i,:] - xyz[:,j,:], axis=1)
        m = np.mean(dij) + 1e-8
        s = np.std(dij)
        scores.append(s/m)
    return float(np.mean(scores))

def features_to_xyz_autolayout(features: np.ndarray, force_layout: str|None):
    cols = strip_height_weight(features)
    N,T,W = cols.shape
    if W % 3 != 0:
        usable = W - (W % 3)
        print(f"[warn] Using first {usable}/{W} columns (multiple of 3).")
        cols = cols[:,:,:usable]
        W = usable
    K = W//3
    if K != 20:
        print(f"[warn] Detected K={K} joints (expected 20). Edges may be wrong.")

    if force_layout in ("interleaved","stacked"):
        layout = force_layout
    else:
        xyz_i = to_xyz_interleaved(cols)
        xyz_s = to_xyz_stacked(cols)
        si = bone_length_score(xyz_i[0], EDGES)
        ss = bone_length_score(xyz_s[0], EDGES)
        layout = "interleaved" if si <= ss else "stacked"
        print(f"[info] Auto-selected layout: {layout} (scores: interleaved={si:.4f}, stacked={ss:.4f})")

    xyz = to_xyz_interleaved(cols) if layout=="interleaved" else to_xyz_stacked(cols)
    return xyz, layout

# ---------- meshcat ----------
def setup_scene(vis, radius: float, K: int):
    sphere = g.Sphere(radius)
    joint_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True)
    for j in range(K):
        vis[f"joints/{j}"].set_object(sphere, joint_mat)
    try:
        vis["/Grid"].set_object(g.GridHelper(size=2.0, divisions=20))
    except Exception:
        pass

def play_realtime_lines(vis, seq_xyz: np.ndarray, fps: int, edges, line_mat):
    """Play a single sequence (T,K,3) in realtime as exact lines."""
    T, K, _ = seq_xyz.shape
    dt = 1.0 / float(fps)
    for t in range(T):
        # move joints
        for j in range(K):
            vis[f"joints/{j}"].set_transform(
                tf.translation_matrix(seq_xyz[t, j].tolist())
            )
        # edges as exact 2-point lines
        for e_idx, (i, j) in enumerate(edges):
            p0 = seq_xyz[t, i]
            p1 = seq_xyz[t, j]
            pts = np.column_stack([p0, p1]).T  # (2,3)
            geom = g.PointsGeometry(pts.T)     # (3,2)
            vis[f"edges/{e_idx}"].set_object(g.Line(geom, line_mat))
        time.sleep(dt)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Visualize 3D keypoints (data/features) with MeshCat.")
    ap.add_argument("--dataset-root", type=str, default=str(Path(__file__).resolve().parent/"datasets"/"dataset0"))
    ap.add_argument("--subject", type=int, help="Subject ID to visualize (required unless --list).")
    ap.add_argument("--seq", type=int, default=0, help="Sequence index within the chosen subject.")
    ap.add_argument("--seq-range", type=str, default=None, help="Play a range 'start:end' (within subject's sequences).")
    ap.add_argument("--all", action="store_true", help="Play all sequences of the subject.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the order of sequences to play.")
    ap.add_argument("--loop", action="store_true", help="Loop the selected sequences until Ctrl+C.")
    ap.add_argument("--pause", type=float, default=0.25, help="Pause (s) between sequences.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--radius", type=float, default=0.02)
    ap.add_argument("--root", type=int, default=midHip, help="Root joint index for recentering (default: midHip=5).")
    ap.add_argument("--layout", choices=["interleaved","stacked"], default=None, help="Force layout; otherwise auto-detect.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    subs = load_subjects(root/"infoData.npy")
    feats = load_features(root/"time_sequences.h5")  # (N, T, F)
    by_subj = index_by_subject(subs)

    if args.list:
        print("Subjects and sequence counts:")
        for sid in sorted(by_subj.keys()):
            print(f"  Subject {sid}: {len(by_subj[sid])} sequences")
        return
    if args.subject is None or args.subject not in by_subj:
        available = sorted(by_subj.keys())
        raise SystemExit(f"--subject required. Available: {available}")

    xyz, layout = features_to_xyz_autolayout(feats, args.layout)  # (N,T,K,3)
    N,T,K,_ = xyz.shape

    # choose which sequences to play for this subject
    seq_ids_all = by_subj[args.subject].tolist()
    if args.all:
        chosen = seq_ids_all
    elif args.seq_range:
        a,b = args.seq_range.split(":")
        a = int(a) if a.strip() else 0
        b = int(b) if b.strip() else len(seq_ids_all)
        chosen = seq_ids_all[a:b]
    else:
        chosen = [by_subj[args.subject][args.seq]]

    if not chosen:
        raise SystemExit("No sequences selected to play.")

    if args.shuffle:
        random.shuffle(chosen)

    vis = meshcat.Visualizer().open()
    print(f"\n[info] Layout used: {layout}. Realtime playback with lines. Ctrl+C to stop.")
    setup_scene(vis, args.radius, K)
    edges = EDGES if K == 20 else [(i, i+1) for i in range(K-1)]
    line_mat = g.LineBasicMaterial(linewidth=2)

    try:
        while True:
            for gi in chosen:
                # per-sequence copy & recenter
                seq = xyz[gi]  # (T,K,3)
                if 0 <= args.root < K:
                    origin = seq[0, args.root:args.root+1, :]
                else:
                    origin = seq[0].mean(axis=0, keepdims=True)
                seq_centered = seq - origin

                print(f"[info] Playing subject {args.subject} sequence global_idx={gi} (T={seq_centered.shape[0]})")
                play_realtime_lines(vis, seq_centered, args.fps, edges, line_mat)
                time.sleep(args.pause)

            if not args.loop:
                break
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize LSTM reconstruction of 3D skeletons with MeshCat.

- White: reference (clean, centered on midHip, height-normalized)
- Red: corrupted input (noise + masking)
- Green: reconstructed from corrupted via trained LSTM

Works with datasets that have either F=50 (-> 48 => K=16) or F=62 (-> 60 => K=20).
You must provide --kmax to match the model you trained (typically 20). The script
pads/truncates sequences to Kmax for inference, but only *renders* the real dataset joints.

Usage
-----
python visualize_reconstruction.py \
  --dataset-root path/to/datasets/datasetX \
  --subject 3 --seq -1 \
  --weights lstm_joint_corrector_best.weights.h5 \
  --model-json lstm_joint_corrector.json \
  --kmax 20
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tfm

# ===============================================================
# Joint config
# ===============================================================

@dataclass
class JointConfig:
    NAMES: List[str] = None
    EDGES: List[Tuple[int, int]] = None

    def __post_init__(self):
        if self.NAMES is None:
            self.NAMES = [
                "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "midHip",
                "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
                "RSmallToe", "LSmallToe", "RBigToe", "LBigToe",
                "RElbow", "LElbow", "RWrist", "LWrist",
            ]
        if self.EDGES is None:
            Neck, RSh, LSh, RHip, LHip, midHip = 0, 1, 2, 3, 4, 5
            RKnee, LKnee, RAnk, LAnk, RHeel, LHeel = 6, 7, 8, 9, 10, 11
            RST, LST, RBT, LBT, RElb, LElb, RWri, LWri = 12, 13, 14, 15, 16, 17, 18, 19
            self.EDGES = [
                (midHip, Neck), (midHip, RHip), (midHip, LHip), (Neck, RSh), (Neck, LSh),
                (RSh, RElb), (RElb, RWri),
                (LSh, LElb), (LElb, LWri),
                (RHip, RKnee), (RKnee, RAnk), (RAnk, RHeel), (RAnk, RST), (RAnk, RBT),
                (RHeel, RST), (RHeel, RBT),
                (LHip, LKnee), (LKnee, LAnk), (LAnk, LHeel), (LAnk, LST), (LAnk, LBT),
                (LHeel, LST), (LHeel, LBT),
            ]

    @property
    def num_joints(self) -> int:
        return len(self.NAMES)

    def get_joint_idx(self, name: str) -> int:
        return self.NAMES.index(name)


# ===============================================================
# Dataset loader & preprocessor
# ===============================================================

class DatasetLoader:
    def __init__(self, dataset_root: Path):
        self.root = Path(dataset_root)
        self.subjects = None
        self.features = None
        self.by_subject = None
        self._load_data()

    def _load_data(self):
        info = np.load(self.root / "infoData.npy", allow_pickle=True)
        if isinstance(info, np.ndarray) and info.dtype == object:
            info = info.item()
        self.subjects = np.asarray(info["subjects"])
        with h5py.File(self.root / "time_sequences.h5", "r") as f:
            self.features = f["data/features"][:]  # (N, T, F)
        self.by_subject = self._index_by_subject()

    def _index_by_subject(self) -> Dict[int, np.ndarray]:
        d = {}
        for i, s in enumerate(self.subjects):
            d.setdefault(int(s), []).append(i)
        for k in d:
            d[k] = np.asarray(d[k], dtype=int)
        return d

    def get_sequences_for_subject(self, subject_id: int) -> np.ndarray:
        if subject_id not in self.by_subject:
            raise ValueError(f"Subject {subject_id} not found. Available: {sorted(self.by_subject.keys())}")
        return self.by_subject[subject_id]


class DataPreprocessor:
    def __init__(self, joint_config: JointConfig):
        self.joint_config = joint_config

    def strip_height_weight(self, features: np.ndarray) -> np.ndarray:
        N, T, F = features.shape
        if F % 3 == 2:
            return features[:, :, : F - 2]
        return features

    def _to_xyz_interleaved(self, cols: np.ndarray) -> np.ndarray:
        N, T, W = cols.shape
        K = W // 3
        return cols.reshape(N, T, K, 3)

    def _to_xyz_stacked(self, cols: np.ndarray) -> np.ndarray:
        N, T, W = cols.shape
        K = W // 3
        xs = cols[:, :, :K]
        ys = cols[:, :, K:2 * K]
        zs = cols[:, :, 2 * K:3 * K]
        return np.stack([xs, ys, zs], axis=-1)

    def _bone_length_score(self, xyz: np.ndarray) -> float:
        T, K, _ = xyz.shape
        scores = []
        for i, j in JointConfig().EDGES:
            if i < K and j < K:
                dij = np.linalg.norm(xyz[:, i, :] - xyz[:, j, :], axis=1)
                m = np.mean(dij) + 1e-8
                s = np.std(dij)
                scores.append(s / m)
        return float(np.mean(scores)) if scores else float("inf")

    def detect_layout(self, features: np.ndarray, force_layout: Optional[str] = None):
        cols = self.strip_height_weight(features)
        N, T, W = cols.shape
        if W % 3 != 0:
            usable = W - (W % 3)
            cols = cols[:, :, :usable]
            W = usable
        if force_layout in ("interleaved", "stacked"):
            layout = force_layout
        else:
            xyz_i = self._to_xyz_interleaved(cols)
            xyz_s = self._to_xyz_stacked(cols)
            si = self._bone_length_score(xyz_i[0])
            ss = self._bone_length_score(xyz_s[0])
            layout = "interleaved" if si <= ss else "stacked"
            print(f"[info] Auto-selected layout: {layout} (scores: interleaved={si:.4f}, stacked={ss:.4f})")
        xyz = self._to_xyz_interleaved(cols) if layout == "interleaved" else self._to_xyz_stacked(cols)
        return xyz, layout

    def subtract_reference_marker(self, xyz: np.ndarray, reference_marker: str = "midHip") -> np.ndarray:
        ref_idx = self.joint_config.get_joint_idx(reference_marker)
        if ref_idx >= xyz.shape[1]:
            # Reference joint missing (dataset with fewer joints): no-op
            return xyz
        ref_value = xyz[:, ref_idx:ref_idx + 1, :]  # (T,1,3)
        return xyz - ref_value

    def normalize_by_height(self, xyz: np.ndarray) -> np.ndarray:
        y_coords = xyz[:, :, 1]
        height = np.max(y_coords) - np.min(y_coords)
        return xyz / max(float(height), 0.1)


# ===============================================================
# Augmenter (build *red* corrupted input stream like training)
# ===============================================================

class DataAugmenter:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def add_noise(self, xyz: np.ndarray, mean_cm: float = 0.0, std_cm: float = 2.0, normalized: bool = True) -> np.ndarray:
        if normalized:
            mean_m = (mean_cm / 100.0) / 1.7
            std_m = (std_cm / 100.0) / 1.7
        else:
            mean_m = mean_cm / 100.0
            std_m = (std_cm / 100.0)
        noise = self.rng.normal(loc=mean_m, scale=max(std_m, 0.0), size=xyz.shape)
        return xyz + noise

    def create_mask(self, T: int, K: int, joint_drop_prob: float = 0.1, frame_drop_frac: float = 0.3, contig_span: Optional[int] = 15) -> np.ndarray:
        mask = np.ones((T, K), dtype=bool)
        n_drop_frames = int(round(frame_drop_frac * T))
        if n_drop_frames > 0:
            if contig_span and contig_span > 1:
                remaining = n_drop_frames
                while remaining > 0:
                    span = min(contig_span, remaining)
                    start = self.rng.integers(0, max(1, T - span + 1))
                    mask[start:start + span, :] = False
                    remaining -= span
            else:
                idx = self.rng.choice(T, size=n_drop_frames, replace=False)
                mask[idx, :] = False
        if joint_drop_prob > 0.0:
            drop_mat = self.rng.random((T, K)) < joint_drop_prob
            mask = mask & (~drop_mat)
        return mask

    def apply_mask(self, xyz: np.ndarray, mask: np.ndarray, mode: str = "nan") -> np.ndarray:
        out = xyz.copy()
        if mode == "nan":
            out[~mask] = np.nan
        else:
            out[~mask] = 0.0
        return out


# ===============================================================
# Model builder / loader
# ===============================================================

SEQ_LEN = 60

def build_joint_model(input_dim: int, seq_len: int = SEQ_LEN, hidden: int = 256, layers_n: int = 3,
                      dropout: float = 0.2, output_dim: Optional[int] = None, l2_coeff: float = 1e-3) -> keras.Model:
    reg = regularizers.l2(l2_coeff)
    inp = keras.Input(shape=(seq_len, input_dim), name="hpe_in")
    x = inp
    for _ in range(layers_n):
        x = layers.Bidirectional(
            layers.LSTM(hidden, return_sequences=True, dropout=dropout,
                        kernel_regularizer=reg, recurrent_regularizer=reg),
            merge_mode="concat",
        )(x)
    x = layers.TimeDistributed(layers.Dense(hidden, activation="relu", kernel_regularizer=reg))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    out_dim = input_dim if output_dim is None else output_dim
    pred_seq = layers.TimeDistributed(layers.Dense(out_dim, activation=None, kernel_regularizer=reg))(x)
    return keras.Model(inp, pred_seq, name="joint_corrector_sequence")

def load_model_for_inference(K_model: int, weights_path: Path, model_json: Optional[Path] = None,
                             hidden: int = 256, layers_n: int = 3, dropout: float = 0.2, l2_coeff: float = 1e-3) -> keras.Model:
    input_dim = K_model * 3
    if model_json is not None and Path(model_json).exists():
        with open(model_json, "r") as f:
            model = keras.models.model_from_json(f.read())
        model.build((None, SEQ_LEN, input_dim))
        model.load_weights(str(weights_path))
        return model
    # Fallback: rebuild with defaults (must match training)
    model = build_joint_model(input_dim=input_dim, seq_len=SEQ_LEN, hidden=hidden,
                              layers_n=layers_n, dropout=dropout, output_dim=input_dim, l2_coeff=l2_coeff)
    model.load_weights(str(weights_path))
    return model


# ===============================================================
# Visualization
# ===============================================================

class SkeletonVisualizer:
    def __init__(self, joint_config: JointConfig, fps: int = 30, radius: float = 0.02):
        self.joint_config = joint_config
        self.fps = fps
        self.radius = radius
        self.vis = meshcat.Visualizer().open()
        self._setup_scene()

    def _setup_scene(self):
        K = self.joint_config.num_joints
        sphere = g.Sphere(self.radius)
        # White reference
        ref_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True)
        for j in range(K):
            self.vis[f"ref/joints/{j}"].set_object(sphere, ref_mat)
        # Red corrupted
        cor_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True, color=0xff0000)
        for j in range(K):
            self.vis[f"cor/joints/{j}"].set_object(sphere, cor_mat)
        # Green reconstructed
        rec_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True, color=0x00ff00)
        for j in range(K):
            self.vis[f"rec/joints/{j}"].set_object(sphere, rec_mat)
        # Grid
        try:
            self.vis["/Grid"].set_object(g.GridHelper(size=2.0, divisions=20))
        except Exception:
            pass

    def _set_line(self, path: str, p1: np.ndarray, p2: np.ndarray, mat: g.LineBasicMaterial):
        # Same approach as your old working code to ensure proper 3D lines
        pts = np.column_stack([p1, p2]).T  # (3,2)
        geom = g.PointsGeometry(pts.T)
        self.vis[path].set_object(g.Line(geom, mat))

    def play_sequence(self, seq_ref: np.ndarray, seq_cor: np.ndarray, mask: np.ndarray, seq_rec: np.ndarray):
        T, K_render, _ = seq_ref.shape
        dt = 1.0 / self.fps
        ref_line_mat = g.LineBasicMaterial(linewidth=2)
        cor_line_mat = g.LineBasicMaterial(linewidth=2, color=0xff0000)
        rec_line_mat = g.LineBasicMaterial(linewidth=2, color=0x00ff00)

        for t in range(T):
            # joints
            for j in range(K_render):
                # reference
                if np.all(np.isfinite(seq_ref[t, j])):
                    self.vis[f"ref/joints/{j}"].set_property("visible", True)
                    self.vis[f"ref/joints/{j}"].set_transform(tfm.translation_matrix(seq_ref[t, j].tolist()))
                else:
                    self.vis[f"ref/joints/{j}"].set_property("visible", False)
                # corrupted (only if visible)
                if mask[t, j] and np.all(np.isfinite(seq_cor[t, j])):
                    self.vis[f"cor/joints/{j}"].set_property("visible", True)
                    self.vis[f"cor/joints/{j}"].set_transform(tfm.translation_matrix(seq_cor[t, j].tolist()))
                else:
                    self.vis[f"cor/joints/{j}"].set_property("visible", False)
                # reconstructed
                if np.all(np.isfinite(seq_rec[t, j])):
                    self.vis[f"rec/joints/{j}"].set_property("visible", True)
                    self.vis[f"rec/joints/{j}"].set_transform(tfm.translation_matrix(seq_rec[t, j].tolist()))
                else:
                    self.vis[f"rec/joints/{j}"].set_property("visible", False)

            # edges
            for e_idx, (i, j) in enumerate(self.joint_config.EDGES):
                if i >= K_render or j >= K_render:
                    continue
                # ref edges
                if np.all(np.isfinite(seq_ref[t, i])) and np.all(np.isfinite(seq_ref[t, j])):
                    self._set_line(f"ref/edges/{e_idx}", seq_ref[t, i], seq_ref[t, j], ref_line_mat)
                else:
                    self.vis[f"ref/edges/{e_idx}"].delete()
                # cor edges
                if (mask[t, i] and mask[t, j] and np.all(np.isfinite(seq_cor[t, i])) and np.all(np.isfinite(seq_cor[t, j]))):
                    self._set_line(f"cor/edges/{e_idx}", seq_cor[t, i], seq_cor[t, j], cor_line_mat)
                else:
                    self.vis[f"cor/edges/{e_idx}"].delete()
                # rec edges
                if np.all(np.isfinite(seq_rec[t, i])) and np.all(np.isfinite(seq_rec[t, j])):
                    self._set_line(f"rec/edges/{e_idx}", seq_rec[t, i], seq_rec[t, j], rec_line_mat)
                else:
                    self.vis[f"rec/edges/{e_idx}"].delete()

            time.sleep(dt)


# ===============================================================
# Helpers: pad/truncate to Kmax for the model, but render only Kseq
# ===============================================================

def pad_or_trunc(seq: np.ndarray, K_model: int) -> np.ndarray:
    """seq: (T,Kseq,3) -> (T,K_model,3)"""
    T, Kseq, _ = seq.shape
    if Kseq == K_model:
        return seq
    if Kseq > K_model:
        return seq[:, :K_model, :]
    out = np.zeros((T, K_model, 3), dtype=seq.dtype)
    out[:, :Kseq, :] = seq
    return out

def pad_or_trunc_mask(mask: np.ndarray, K_model: int) -> np.ndarray:
    """mask: (T,Kseq) -> (T,K_model)"""
    T, Kseq = mask.shape
    if Kseq == K_model:
        return mask
    if Kseq > K_model:
        return mask[:, :K_model]
    out = np.zeros((T, K_model), dtype=mask.dtype)
    out[:, :Kseq] = mask
    return out


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize LSTM reconstruction on dataset (handles K=16 or K=20).")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seq", type=int, default=-1, help="Sequence index; -1 to loop over all sequences")
    parser.add_argument("--weights", type=str, required=True, help="Path to .weights.h5 checkpoint")
    parser.add_argument("--model-json", type=str, default=None, help="Optional JSON architecture file")
    parser.add_argument("--kmax", type=int, default=20, help="Model Kmax used during training (e.g., 20)")

    # viz
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=float, default=0.02)

    # corrupted red stream (same style as training)
    parser.add_argument("--noise-mean-cm", type=float, default=0.0)
    parser.add_argument("--noise-std-cm", type=float, default=2.0)
    parser.add_argument("--joint-drop-prob", type=float, default=0.05)
    parser.add_argument("--frame-drop-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Data
    joint_cfg = JointConfig()
    loader = DatasetLoader(args.dataset_root)
    pre = DataPreprocessor(joint_cfg)
    xyz_all, layout = pre.detect_layout(loader.features)

    seq_ids = loader.get_sequences_for_subject(args.subject)
    T_detected, K_detected, _ = xyz_all[seq_ids[0]].shape
    if T_detected != SEQ_LEN:
        print(f"[warn] Expected T={SEQ_LEN}, got {T_detected}. Proceeding anyway.")
    print(f"Dataset K={K_detected}, model Kmax={args.kmax}")

    # Model (fixed Kmax as used in training)
    model = load_model_for_inference(
        K_model=args.kmax,
        weights_path=Path(args.weights),
        model_json=Path(args.model_json) if args.model_json else None
    )

    vis = SkeletonVisualizer(joint_cfg, fps=args.fps, radius=args.radius)
    print(f"Layout: {layout}")
    print("Visualization running (Ctrl+C to stop)...")

    aug = DataAugmenter(seed=args.seed)

    def process_one(seq_raw: np.ndarray):
        """
        seq_raw: (T, Kseq, 3) from dataset.
        Returns data for rendering limited to Kseq (no padded joints shown).
        """
        T_, Kseq, _ = seq_raw.shape
        # 1) Reference clean
        seq_ref = pre.subtract_reference_marker(seq_raw.copy(), "midHip")
        seq_ref = pre.normalize_by_height(seq_ref)

        # 2) Build corrupted stream for viz + model input
        seq_cor = aug.add_noise(seq_ref.copy(), args.noise_mean_cm, args.noise_std_cm, normalized=True)
        mask = aug.create_mask(T=T_, K=Kseq, joint_drop_prob=args.joint_drop_prob, frame_drop_frac=args.frame_drop_frac)

        # Red display: NaN where dropped
        seq_cor_vis = aug.apply_mask(seq_cor.copy(), mask, mode="nan")

        # Model input: zero where dropped + pad/trunc to Kmax
        seq_cor_in = seq_cor.copy()
        seq_cor_in[~mask] = 0.0
        seq_cor_in_pad = pad_or_trunc(seq_cor_in, args.kmax)
        x_in = seq_cor_in_pad.reshape(1, T_, args.kmax * 3).astype(np.float32)

        # 3) Predict and crop to real Kseq for rendering
        y_pred_pad = model.predict(x_in, verbose=0).reshape(T_, args.kmax, 3)
        y_pred = y_pred_pad[:, :Kseq, :]

        return seq_ref, seq_cor_vis, mask, y_pred

    try:
        while True:
            if args.seq >= 0:
                idx = seq_ids[args.seq]
                seq = xyz_all[idx].copy()
                seq_ref, seq_cor_vis, mask, seq_rec = process_one(seq)
                vis.play_sequence(seq_ref, seq_cor_vis, mask, seq_rec)
            else:
                for local_i, idx in enumerate(seq_ids):
                    print(f"Sequence {local_i+1}/{len(seq_ids)}")
                    seq = xyz_all[idx].copy()
                    seq_ref, seq_cor_vis, mask, seq_rec = process_one(seq)
                    vis.play_sequence(seq_ref, seq_cor_vis, mask, seq_rec)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()

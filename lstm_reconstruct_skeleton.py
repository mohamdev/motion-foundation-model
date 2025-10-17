#!/usr/bin/env python3
"""
Train an LSTM to reconstruct clean, height-normalized 3D skeleton sequences
from corrupted ones (noise + missing joints), across one or more dataset folders.

Multi-dataset
-------------
- Set --datasets_root to the parent folder that contains dataset subfolders
  like dataset0, dataset1, etc.
- Use --datasets -1 to include ALL dataset* folders (after filtering).
- Use --datasets N  to include the FIRST N dataset* folders (natural order), then
  filter to those with total F==62 (60 joints + 2 height/weight).

Memory/perf features
--------------------
- --mixed-precision: enable AMP (float16 math on GPU, fp32 output head).
- LSTM uses cuDNN fast path (no recurrent dropout; dropout moved after each block).
- --accum-steps: gradient accumulation to keep effective batch size large.
- --merge-mode {sum,concat}: choose BiLSTM merge (default=sum to save memory).

Key points
----------
- Height-normalized inputs/targets centered on midHip.
- Inputs are corrupted (noise + masking); missing joints zero-filled.
- Loss masking by timestep (fraction of visible joints).
- Loss = MSE + optional L2.
- Saves model JSON + weights (.weights.h5).
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Optional: avoid pre-grabbing all VRAM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("[warn] set_memory_growth failed:", e)

# ===============================================================
# Configuration for joints (must match the dataset layout)
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
# Dataset loader (supports multiple dataset* folders) — FILTER F==62
# ===============================================================

def _natural_key(p: Path) -> Tuple[int, str]:
    import re
    m = re.search(r"(\d+)$", p.name)
    return (int(m.group(1)) if m else 10**9, p.name)

def probe_dataset_shape(root: Path) -> Tuple[int, int, int]:
    with h5py.File(root / "time_sequences.h5", "r") as f:
        dset = f["data/features"]
        N, T, F = dset.shape
    return int(N), int(T), int(F)

class MultiDatasetLoader:
    def __init__(self, dataset_roots: List[Path], require_total_F: int = 62):
        self.roots = dataset_roots
        self.require_total_F = require_total_F
        self.subjects = None
        self.features = None
        self.by_subject = None
        self._load_all()

    def _load_one(self, root: Path):
        info = np.load(root / "infoData.npy", allow_pickle=True)
        if isinstance(info, np.ndarray) and info.dtype == object:
            info = info.item()
        subjects = np.asarray(info["subjects"])
        with h5py.File(root / "time_sequences.h5", "r") as f:
            feats = f["data/features"][:]  # (N, T, F)
        return subjects, feats

    def _load_all(self):
        kept, skipped = [], []
        for r in self.roots:
            try:
                N, T, F = probe_dataset_shape(r)
                (kept if F == self.require_total_F else skipped).append((r, N, T, F))
            except Exception:
                skipped.append((r, -1, -1, -1))
        print("Dataset scan:")
        for r, N, T, F in kept:
            print(f"  KEEP   {r.name}: N={N}, T={T}, F={F}")
        for r, N, T, F in skipped:
            reason = "I/O error" if F == -1 else f"F={F} != {self.require_total_F}"
            print(f"  SKIP   {r.name}: {reason}")
        if not kept:
            raise RuntimeError(f"No datasets with total F=={self.require_total_F} found among selected roots.")
        base_T = kept[0][2]
        badT = [r for (r, _, T, _) in kept if T != base_T]
        if badT:
            names = ", ".join(p.name for p in badT)
            raise ValueError(f"Inconsistent T across kept datasets (expected {base_T}): {names}")

        all_subj, all_feat = [], []
        print("Loading kept datasets:")
        for r, _, T, F in kept:
            s, f = self._load_one(r)
            assert f.shape[1] == T and f.shape[2] == F
            print(f"  - {r.name}: N={len(s)}, T={f.shape[1]}, F={f.shape[2]}")
            all_subj.append(s); all_feat.append(f)
        self.subjects = np.concatenate(all_subj, axis=0)
        self.features = np.concatenate(all_feat, axis=0)
        self.by_subject = self._index_by_subject()
        print(f"Total (kept only): N={len(self.subjects)}, T={self.features.shape[1]}, F={self.features.shape[2]}")

    def _index_by_subject(self) -> Dict[int, np.ndarray]:
        d = {}
        for i, s in enumerate(self.subjects):
            d.setdefault(int(s), []).append(i)
        for k in d:
            d[k] = np.asarray(d[k], dtype=int)
        return d


# ===============================================================
# Preprocessor
# ===============================================================

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
        for i, j in self.joint_config.EDGES:
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
            cols = cols[:, :, : (W - (W % 3))]
            W = cols.shape[2]
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
        ref_value = xyz[:, ref_idx:ref_idx + 1, :]  # (T,1,3)
        return xyz - ref_value

    def estimate_height(self, xyz: np.ndarray) -> float:
        y_coords = xyz[:, :, 1]
        height = np.max(y_coords) - np.min(y_coords)
        return max(float(height), 0.1)

    def normalize_by_height(self, xyz: np.ndarray, height: Optional[float] = None) -> np.ndarray:
        if height is None:
            height = self.estimate_height(xyz)
        return xyz / height


# ===============================================================
# Corruption utilities (noise + masking)
# ===============================================================

class Corruptor:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def add_noise(self, xyz: np.ndarray, mean_cm: float = 0.0, std_cm: float = 2.0) -> np.ndarray:
        mean_m = (mean_cm / 100.0) / 1.7
        std_m = (std_cm / 100.0) / 1.7
        noise = self.rng.normal(loc=mean_m, scale=max(std_m, 0.0), size=xyz.shape)
        return xyz + noise

    def create_mask(self, T: int, K: int, joint_drop_prob: float = 0.1, frame_drop_frac: float = 0.3,
                    contig_span: Optional[int] = 15) -> np.ndarray:
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


# ===============================================================
# Model (+ AMP safe output) and gradient accumulation
# ===============================================================

SEQ_LEN_DEFAULT = 60  # dataset sequences are length 60 by construction

def build_joint_model(input_dim: int, seq_len: int, hidden: int = 256, layers_n: int = 3,
                      dropout: float = 0.2, output_dim: Optional[int] = None, l2_coeff: float = 1e-3,
                      merge_mode: str = "concat", post_dropout: float = 0.0) -> keras.Model:
    reg = regularizers.l2(l2_coeff)
    inp = keras.Input(shape=(seq_len, input_dim), name="hpe_in")
    x = inp

    for _ in range(layers_n):
        x = layers.Bidirectional(
            layers.LSTM(
                hidden, return_sequences=True,
                dropout=dropout,                   # keep in-LSTM dropout
                kernel_regularizer=reg, recurrent_regularizer=reg
            ),
            merge_mode=merge_mode                 # "concat" restores capacity
        )(x)
        if post_dropout and post_dropout > 0.0:
            x = layers.Dropout(post_dropout)(x)   # turn off by default

    x = layers.TimeDistributed(layers.Dense(hidden, activation="relu", kernel_regularizer=reg))(x)
    # (optional) no extra dropout here to keep capacity high

    out_dim = input_dim if output_dim is None else output_dim
    pred_seq = layers.TimeDistributed(layers.Dense(out_dim, activation=None, kernel_regularizer=reg))(x)

    # Cast to float32 for numerically stable loss in mixed precision
    pred_seq = layers.Activation("linear", dtype="float32", name="cast_fp32")(pred_seq)

    return keras.Model(inp, pred_seq, name="joint_corrector_sequence")


class AccumModel(keras.Model):
    """Gradient accumulation wrapper: dataset should yield (x, y, sample_weight[B,T])."""
    def __init__(self, accum_steps=1, **kwargs):
        super().__init__(**kwargs)
        self.accum_steps = int(max(1, accum_steps))
        self._grad_accum = None
        self._step = tf.Variable(0, trainable=False, dtype=tf.int32, name="accum_step")

    @tf.function  # make it graph-friendly
    def train_step(self, data):
        x, y, w = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Keras will handle loss reduction and sample_weight correctly.
            loss = self.compute_loss(x, y, y_pred, sample_weight=w)

        grads = tape.gradient(loss, self.trainable_variables)
        # Replace None grads with zeros of the same shape as variables
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, self.trainable_variables)]

        # Lazily create accumulators matching variable shapes/dtypes
        if self._grad_accum is None:
            self._grad_accum = [tf.Variable(tf.zeros_like(v), trainable=False)
                                for v in self.trainable_variables]

        # Accumulate
        for ga, g in zip(self._grad_accum, grads):
            ga.assign_add(g)

        # Increment step and decide (tensor-wise) if we apply now
        step = self._step.assign_add(1)
        apply_now = tf.equal(tf.math.floormod(step, self.accum_steps), 0)

        def _apply():
            # Average grads so this matches a true larger batch
            denom = tf.cast(self.accum_steps, self._grad_accum[0].dtype)
            avg_grads = [ga / denom for ga in self._grad_accum]
            self.optimizer.apply_gradients(zip(avg_grads, self.trainable_variables))
            # Reset accumulators
            for ga in self._grad_accum:
                ga.assign(tf.zeros_like(ga))
            return 0  # tf.cond needs a return

        # Graph-safe conditional
        tf.cond(apply_now, _apply, lambda: 0)

        # Metrics (includes loss)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=w)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs



# ===============================================================
# Data pipeline (tf.data) with masked loss
# ===============================================================

def make_sequence_pairs(xyz_data: np.ndarray, seq_indices: np.ndarray, joint_config: JointConfig,
                        noise_mean_cm: float, noise_std_cm: float, joint_drop_prob: float, frame_drop_frac: float,
                        seed: Optional[int] = None, reference_marker: str = "midHip"):
    T = xyz_data.shape[1]
    K = xyz_data.shape[2]
    corrupt = Corruptor(seed)

    pairs = []
    for idx in seq_indices:
        seq = xyz_data[idx].copy()  # (T, K, 3)
        ref_idx = joint_config.get_joint_idx(reference_marker)
        ref_traj = seq[:, ref_idx:ref_idx + 1, :]
        seq = seq - ref_traj
        y_coords = seq[:, :, 1]
        height = max(float(np.max(y_coords) - np.min(y_coords)), 0.1)
        clean = seq / height
        noisy = corrupt.add_noise(clean, mean_cm=noise_mean_cm, std_cm=noise_std_cm)
        mask = corrupt.create_mask(T=T, K=K, joint_drop_prob=joint_drop_prob, frame_drop_frac=frame_drop_frac)
        inp = noisy.copy(); inp[~mask] = 0.0
        clean_f = clean.reshape(T, -1).astype(np.float32)
        inp_f = inp.reshape(T, -1).astype(np.float32)
        mask_time = mask.mean(axis=1).astype(np.float32)  # (T,)
        pairs.append((inp_f, clean_f, mask_time))
    return pairs


def tf_dataset_from_pairs(pairs, batch_size: int, shuffle: bool = True, buffer_size: int = 2048):
    x = np.stack([p[0] for p in pairs], axis=0)
    y = np.stack([p[1] for p in pairs], axis=0)
    w = np.stack([p[2] for p in pairs], axis=0)  # (B, T)
    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    if shuffle:
        ds = ds.shuffle(min(len(pairs), buffer_size), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ===============================================================
# Train / Eval
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Train LSTM to reconstruct clean skeletons from corrupted inputs (multi-dataset, F==62 only).")
    parser.add_argument("--datasets_root", type=str, required=True)
    parser.add_argument("--datasets", type=int, default=-1)
    parser.add_argument("--reference-marker", type=str, default="midHip")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--noise-mean-cm", type=float, default=0.0)
    parser.add_argument("--noise-std-cm", type=float, default=2.0)
    parser.add_argument("--joint-drop-prob", type=float, default=0.0)
    parser.add_argument("--frame-drop-frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--save-prefix", type=str, default="lstm_joint_corrector")

    # New flags
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable AMP (float16 compute on GPU, fp32 output head)")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Accumulate gradients over this many steps (keeps effective batch large)")
    parser.add_argument("--merge-mode", type=str, default="concat", choices=["sum", "concat"],
                        help="BiLSTM merge mode; 'sum' halves activation size vs 'concat'")

    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Mixed precision (must be set before creating the model/optimizer)
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[info] Mixed precision enabled (policy=mixed_float16)")

    # Discover dataset* folders
    root = Path(args.datasets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"datasets_root does not exist: {root}")
    candidates = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("dataset")], key=_natural_key)
    if len(candidates) == 0:
        raise RuntimeError(f"No dataset* folders found under {root}")
    selected = candidates if args.datasets == -1 else candidates[:args.datasets] if args.datasets > 0 else (_ for _ in ()).throw(ValueError("--datasets must be -1 or a positive integer"))
    print("Selected (pre-filter):", ", ".join(p.name for p in selected))

    # Load (filter to F==62) & merge
    joint_cfg = JointConfig()
    loader = MultiDatasetLoader(selected, require_total_F=62)
    pre = DataPreprocessor(joint_cfg)

    # Convert to XYZ (strip height/weight)
    xyz_all, layout = pre.detect_layout(loader.features)
    N, T, K, _ = xyz_all.shape
    assert T == args.seq_len, f"Expected sequence length {args.seq_len}, but got {T}."
    print(f"Loaded (merged, F==62 only): N={N} sequences, T={T} frames, K={K} joints (layout={layout})")

    # Subject split
    unique_subjects = sorted(set(int(s) for s in loader.subjects))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_subjects)
    n_val = max(1, int(round(len(unique_subjects) * args.val_split)))
    val_subjects = set(unique_subjects[:n_val])
    train_subjects = set(unique_subjects[n_val:])
    train_idx = np.array([i for i, s in enumerate(loader.subjects) if int(s) in train_subjects], dtype=int)
    val_idx = np.array([i for i, s in enumerate(loader.subjects) if int(s) in val_subjects], dtype=int)
    print(f"Train subjects: {sorted(train_subjects)} (n_seq={len(train_idx)})")
    print(f"Val subjects:   {sorted(val_subjects)} (n_seq={len(val_idx)})")

    # Build pairs
    print("Preparing training pairs...")
    train_pairs = make_sequence_pairs(
        xyz_all, train_idx, joint_cfg,
        noise_mean_cm=args.noise_mean_cm, noise_std_cm=args.noise_std_cm,
        joint_drop_prob=args.joint_drop_prob, frame_drop_frac=args.frame_drop_frac,
        seed=args.seed, reference_marker=args.reference_marker,
    )
    print("Preparing validation pairs...")
    val_pairs = make_sequence_pairs(
        xyz_all, val_idx, joint_cfg,
        noise_mean_cm=args.noise_mean_cm, noise_std_cm=args.noise_std_cm,
        joint_drop_prob=args.joint_drop_prob, frame_drop_frac=args.frame_drop_frac,
        seed=args.seed + 1, reference_marker=args.reference_marker,
    )

    input_dim = K * 3
    base_model = build_joint_model(
        input_dim=input_dim,
        seq_len=T,
        hidden=args.hidden,
        layers_n=args.layers,
        dropout=args.dropout,
        output_dim=input_dim,
        l2_coeff=args.l2,
        merge_mode=args.merge_mode,
    )
    # Wrap for accumulation if needed
    model = base_model if args.accum_steps <= 1 else AccumModel(accum_steps=args.accum_steps,
                                                                inputs=base_model.inputs,
                                                                outputs=base_model.outputs,
                                                                name=base_model.name)
    model.summary()

    # Optimizer (AMP automatically applies loss scaling when policy is mixed_float16)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss="mse")


    # tf.data datasets
    ds_train = tf_dataset_from_pairs(train_pairs, batch_size=args.batch_size, shuffle=True)
    ds_val = tf_dataset_from_pairs(val_pairs, batch_size=args.batch_size, shuffle=False)

    # Callbacks
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=f"{args.save_prefix}_best.weights.h5",
        monitor="val_loss", save_best_only=True, save_weights_only=True
    )
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1)


    # Train
    history = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=[ckpt, early, reduce])

    # Evaluate masked MSE explicitly
    def eval_masked(ds):
        tot_num = 0.0
        tot_den = 0.0
        for xb, yb, wb in ds:
            yp = model(xb, training=False)
            sq = tf.square(yb - yp) * tf.expand_dims(wb, axis=-1)
            tot_num += tf.reduce_sum(sq).numpy()
            tot_den += (tf.reduce_sum(wb).numpy() * yb.shape[-1] + 1e-8)
        return tot_num / tot_den

    train_mse = eval_masked(ds_train)
    val_mse = eval_masked(ds_val)
    print(f"Masked MSE — train: {train_mse:.6f} | val: {val_mse:.6f}")

    # Save model JSON + weights
    json_path = f"{args.save_prefix}.json"
    weights_path = f"{args.save_prefix}.weights.h5"
    with open(json_path, "w") as f:
        f.write(model.to_json())
    model.save_weights(weights_path)
    print(f"Saved model architecture to {json_path} and weights to {weights_path}")


if __name__ == "__main__":
    main()

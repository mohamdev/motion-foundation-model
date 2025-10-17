#!/usr/bin/env python3
"""
Visualize continuous motion with centroidal dynamics overlay.

This script visualizes:
- Skeleton motion (continuous, reconstructed from sliding windows)
- Center of mass (red sphere)
- Linear momentum (blue arrow from midHip)
- Angular momentum (green spinning disc/arrow about COM)

Usage
-----
python visualize_continuous_motion_with_dynamics.py \
  --dataset-root ./datasets/dataset0 \
  --subject 3 \
  --fps 30
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

from centroidal_dynamics import CentroidalDynamics, extract_height_weight_from_features

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


def load_labels(h5_path: Path) -> Optional[np.ndarray]:
    """Load marker data (labels) if available."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "data/labels" in f:
                return f["data/labels"][:]
    except Exception:
        pass
    return None


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
    """Detect the stride of sliding window by comparing consecutive sequences."""
    if len(sequences) < 2:
        return sequences.shape[1]

    seq1 = sequences[0]
    seq2 = sequences[1]
    T = seq1.shape[0]

    for offset in range(1, T):
        if np.allclose(seq1[offset], seq2[0], atol=1e-6):
            print(f"[info] Detected sliding window stride: {offset} frames")
            return offset

    print(f"[info] No overlap detected - using stride of {T} frames")
    return T


def reconstruct_continuous_motion(sequences: np.ndarray, stride: Optional[int] = None) -> np.ndarray:
    """Reconstruct continuous motion from sliding window sequences."""
    N, T, K, _ = sequences.shape

    if N == 1:
        return sequences[0]

    if stride is None:
        stride = detect_sliding_window_stride(sequences)

    total_length = T + (N - 1) * stride
    continuous = np.zeros((total_length, K, 3), dtype=sequences.dtype)
    weight = np.zeros((total_length, K, 3), dtype=np.float32)

    for i, seq in enumerate(sequences):
        start_idx = i * stride
        end_idx = start_idx + T
        continuous[start_idx:end_idx] += seq
        weight[start_idx:end_idx] += 1.0

    mask = weight > 0
    continuous[mask] /= weight[mask]

    print(f"[info] Reconstructed continuous motion: {N} sequences × {T} frames → {total_length} frames")
    return continuous


# ===============================================================
# Visualization with dynamics
# ===============================================================

def setup_scene_with_dynamics(vis, radius: float, K: int, num_markers: int = 0):
    """Setup meshcat scene with skeleton and dynamics visualization objects."""
    # Skeleton joints
    sphere = g.Sphere(radius)
    joint_mat = g.MeshLambertMaterial(opacity=0.95, transparent=True, color=0xffffff)
    for j in range(K):
        vis[f"skeleton/joints/{j}"].set_object(sphere, joint_mat)

    # Markers (smaller, light gray)
    if num_markers > 0:
        marker_sphere = g.Sphere(radius * 0.3)
        marker_mat = g.MeshLambertMaterial(opacity=0.7, transparent=True, color=0xcccccc)
        for m in range(num_markers):
            vis[f"markers/{m}"].set_object(marker_sphere, marker_mat)

    # Center of mass (larger red sphere)
    com_sphere = g.Sphere(radius * 2.5)
    com_mat = g.MeshLambertMaterial(opacity=0.9, transparent=True, color=0xff0000)
    vis["dynamics/com"].set_object(com_sphere, com_mat)

    # Grid
    try:
        vis["/Grid"].set_object(g.GridHelper(size=2.0, divisions=20))
    except Exception:
        pass


def create_arrow_mesh(length: float, radius: float = 0.01, color: int = 0x0000ff):
    """
    Create an arrow geometry for visualizing vectors.

    The arrow points along the +Z axis by default, then we rotate it.
    """
    # Note: We'll use a cylinder + cone for the arrow
    # This is a simplified version - meshcat will handle the positioning
    pass


def set_arrow(vis, path: str, start: np.ndarray, direction: np.ndarray,
              color: int = 0x0000ff, line_width: float = 30.0):
    """
    Set an arrow from start point in given direction using lines.

    Args:
        vis: Meshcat visualizer
        path: Path in scene tree
        start: (3,) starting position
        direction: (3,) direction vector (magnitude = arrow length)
        color: Arrow color
        line_width: Width of the arrow line
    """
    magnitude = np.linalg.norm(direction)

    if magnitude < 1e-6:
        # Hide arrow if magnitude too small
        vis[path].set_property("visible", False)
        return

    vis[path].set_property("visible", True)

    # End point of arrow
    end = start + direction

    # Create arrow shaft as a line from start to end
    line_mat = g.LineBasicMaterial(linewidth=line_width, color=color)
    shaft_points = np.column_stack([start, end]).T  # (2, 3)
    shaft_geom = g.PointsGeometry(shaft_points.T)  # (3, 2)
    vis[f"{path}/shaft"].set_object(g.Line(shaft_geom, line_mat))

    # Create arrow head (two lines forming a V shape)
    # Head is 20% of arrow length
    head_length = magnitude * 0.2
    head_width = magnitude * 0.1

    if magnitude > 1e-6:
        # Direction vector
        dir_norm = direction / magnitude

        # Create perpendicular vectors for the arrow head
        # Find a vector perpendicular to direction
        if abs(dir_norm[0]) < 0.9:
            perp1 = np.cross(dir_norm, [1, 0, 0])
        else:
            perp1 = np.cross(dir_norm, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(dir_norm, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Arrow head points
        head_base = end - dir_norm * head_length
        head_point1 = head_base + perp1 * head_width
        head_point2 = head_base + perp2 * head_width
        head_point3 = head_base - perp1 * head_width
        head_point4 = head_base - perp2 * head_width

        # Draw 4 lines from head points to arrow tip
        for i, head_point in enumerate([head_point1, head_point2, head_point3, head_point4]):
            head_line_points = np.column_stack([head_point, end]).T
            head_line_geom = g.PointsGeometry(head_line_points.T)
            vis[f"{path}/head{i}"].set_object(g.Line(head_line_geom, line_mat))


def align_vector_with_z(direction: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create transformation matrix that aligns Z-axis with given direction.

    Args:
        direction: (3,) unit direction vector
        translation: (3,) translation vector

    Returns:
        4x4 transformation matrix
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Create rotation matrix to align [0,0,1] with direction
    z_axis = np.array([0.0, 0.0, 1.0])

    # If direction is already along Z, return identity rotation
    if np.allclose(direction, z_axis):
        return tf.translation_matrix(translation.tolist())

    # If direction is opposite to Z
    if np.allclose(direction, -z_axis):
        rot = tf.rotation_matrix(np.pi, [1, 0, 0])
        rot[:3, 3] = translation
        return rot

    # General case: use Rodrigues' rotation formula
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    s = np.linalg.norm(v)

    # Skew-symmetric matrix
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    # Rotation matrix
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s + 1e-10))

    # Create 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T


def set_angular_momentum_visualization(vis, path: str, com: np.ndarray,
                                      angular_momentum: np.ndarray,
                                      scale: float = 1.0):
    """
    Visualize angular momentum as a spinning disc with axis arrow.

    Args:
        vis: Meshcat visualizer
        path: Path in scene tree
        com: (3,) center of mass position
        angular_momentum: (3,) angular momentum vector
        scale: Scaling factor for visualization
    """
    magnitude = np.linalg.norm(angular_momentum)

    if magnitude < 1e-6:
        vis[path].set_property("visible", False)
        return

    vis[path].set_property("visible", True)

    # Direction of angular momentum (axis of rotation)
    axis = angular_momentum / magnitude

    # Create a disc perpendicular to the angular momentum
    disc_radius = min(0.15, magnitude * scale * 0.1)
    disc_thickness = 0.005
    disc = g.Cylinder(disc_thickness, disc_radius)
    disc_mat = g.MeshLambertMaterial(color=0x00ff00, opacity=0.6, transparent=True)

    vis[f"{path}/disc"].set_object(disc, disc_mat)

    # Orient disc perpendicular to angular momentum axis
    # Cylinder is along Z, we want it perpendicular to 'axis'
    # So we align Z with axis
    disc_transform = align_vector_with_z(axis, com)
    vis[f"{path}/disc"].set_transform(disc_transform)

    # Add arrow along angular momentum direction
    arrow_length = magnitude * scale * 0.5
    set_arrow(vis, f"{path}/arrow", com, axis * arrow_length,
              color=0x00ff00, line_width=30.0)


def play_continuous_motion_with_dynamics(vis, motion: np.ndarray, dynamics: CentroidalDynamics,
                                        fps: int, edges: List[Tuple[int, int]],
                                        reference_joint: int = midHip,
                                        momentum_scale: float = 0.1,
                                        angular_momentum_scale: float = 1.0,
                                        markers: Optional[np.ndarray] = None):
    """
    Play continuous motion with centroidal dynamics visualization.

    Args:
        vis: MeshCat visualizer
        motion: (T, K, 3) continuous motion array
        dynamics: CentroidalDynamics instance
        fps: Frames per second
        edges: List of skeleton edges
        reference_joint: Joint to use as reference for linear momentum arrow
        momentum_scale: Scale factor for linear momentum arrows
        angular_momentum_scale: Scale factor for angular momentum visualization
    """
    T, K, _ = motion.shape
    dt = 1.0 / float(fps)
    line_mat = g.LineBasicMaterial(linewidth=2, color=0xaaaaaa)

    # Center skeleton on first frame's reference joint (BEFORE rotation)
    if 0 <= reference_joint < K:
        skeleton_origin = motion[0, reference_joint:reference_joint+1, :]
    else:
        skeleton_origin = motion[0].mean(axis=0, keepdims=True)

    motion_centered = motion - skeleton_origin

    # Center markers on their own hip position (to align with skeleton)
    if markers is not None:
        # Marker indices: RHJC=41, LHJC=42 (Right/Left Hip Joint Centers)
        # midHip for markers = average of RHJC and LHJC
        RHJC_idx = 41
        LHJC_idx = 42

        if markers.shape[1] > LHJC_idx:  # Check if we have enough markers
            # Calculate midHip from hip joint center markers
            RHJC = markers[0, RHJC_idx:RHJC_idx+1, :]  # First frame, right hip
            LHJC = markers[0, LHJC_idx:LHJC_idx+1, :]  # First frame, left hip
            markers_origin = (RHJC + LHJC) / 2.0  # midHip
            print(f"[info] Centering markers on midHip (avg of RHJC and LHJC)")
        else:
            # Fallback: use mean if we don't have RHJC/LHJC
            markers_origin = markers[0].mean(axis=0, keepdims=True)
            print(f"[warn] RHJC/LHJC markers not found, using mean position")

        markers_centered = markers - markers_origin
    else:
        markers_centered = None

    # Compute all centroidal dynamics
    print(f"[info] Computing centroidal dynamics...")
    all_dynamics = dynamics.compute_all_dynamics(motion)

    com = all_dynamics['com'] - skeleton_origin  # Center COM using skeleton origin
    com_velocity = all_dynamics['com_velocity']
    linear_momentum = all_dynamics['linear_momentum']
    angular_momentum = all_dynamics['angular_momentum']

    print(f"[info] Playing continuous motion: {T} frames at {fps} fps ({T/fps:.1f} seconds)")

    # Prepare markers if provided (apply same rotation as skeleton)
    if markers_centered is not None:
        markers_rotated = markers_centered.copy()
        markers_rotated[..., 0] = markers_centered[..., 0]
        markers_rotated[..., 1] = -markers_centered[..., 2]
        markers_rotated[..., 2] = markers_centered[..., 1]
        num_markers = markers.shape[1]
        print(f"[info] Visualizing {num_markers} markers")
    else:
        markers_rotated = None
        num_markers = 0

    for t in range(T):
        # Update skeleton joints
        for j in range(K):
            vis[f"skeleton/joints/{j}"].set_transform(
                tf.translation_matrix(motion_centered[t, j].tolist())
            )

        # Update markers
        if markers_rotated is not None:
            for m in range(num_markers):
                vis[f"markers/{m}"].set_transform(
                    tf.translation_matrix(markers_rotated[t, m].tolist())
                )

        # Update skeleton edges
        for e_idx, (i, j) in enumerate(edges):
            if i < K and j < K:
                p0 = motion_centered[t, i]
                p1 = motion_centered[t, j]
                pts = np.column_stack([p0, p1]).T
                geom = g.PointsGeometry(pts.T)
                vis[f"skeleton/edges/{e_idx}"].set_object(g.Line(geom, line_mat))

        # Update center of mass
        vis["dynamics/com"].set_transform(
            tf.translation_matrix(com[t].tolist())
        )

        # Update linear momentum arrow (from midHip, pointing in COM velocity direction)
        # Linear momentum direction = COM velocity direction (p = m*v, same direction as v)
        midhip_pos = motion_centered[t, midHip]
        # Scale by momentum magnitude for visual representation
        momentum_magnitude = np.linalg.norm(linear_momentum[t])
        velocity_magnitude = np.linalg.norm(com_velocity[t])
        if velocity_magnitude > 1e-6:
            # Use velocity direction, but scale by momentum for arrow length
            velocity_direction = com_velocity[t] / velocity_magnitude
            momentum_direction = velocity_direction * momentum_magnitude * momentum_scale
        else:
            momentum_direction = np.zeros(3)
        set_arrow(vis, "dynamics/linear_momentum", midhip_pos, momentum_direction,
                 color=0x0000ff, line_width=30.0)

        # Update angular momentum visualization
        set_angular_momentum_visualization(
            vis, "dynamics/angular_momentum", com[t],
            angular_momentum[t], scale=angular_momentum_scale
        )

        time.sleep(dt)


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize continuous motion with centroidal dynamics"
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=float, default=0.02)
    parser.add_argument("--layout", choices=["interleaved", "stacked"], default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--momentum-scale", type=float, default=0.03,
                       help="Scale factor for linear momentum arrows")
    parser.add_argument("--angular-momentum-scale", type=float, default=1.0,
                       help="Scale factor for angular momentum visualization")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    # Load data
    root = Path(args.dataset_root)
    subjects = load_subjects(root / "infoData.npy")
    features = load_features(root / "time_sequences.h5")
    labels = load_labels(root / "time_sequences.h5")  # Marker data
    by_subject = index_by_subject(subjects)

    if args.list:
        print("Available subjects:")
        for sid in sorted(by_subject.keys()):
            n_seq = len(by_subject[sid])
            print(f"  Subject {sid}: {n_seq} sequences")
        return

    if args.subject not in by_subject:
        available = sorted(by_subject.keys())
        raise SystemExit(f"Subject {args.subject} not found. Available: {available}")

    # Get sequences for this subject
    seq_indices = by_subject[args.subject]
    subject_features = features[seq_indices]
    subject_labels = labels[seq_indices] if labels is not None else None

    # Extract height and weight
    height, weight = extract_height_weight_from_features(subject_features[0])
    print(f"\n[info] Subject {args.subject}: height={height:.3f}m, weight={weight:.3f}kg")

    # Convert to XYZ
    xyz_all, layout = features_to_xyz_autolayout(subject_features, args.layout)
    sequences = xyz_all
    N, T, K, _ = sequences.shape

    print(f"[info] {N} sequences of {T} frames each, {K} joints")

    # Process markers if available
    if subject_labels is not None:
        # Markers are in labels, should be (N, T, M*3) where M is number of markers
        N_lab, T_lab, F_lab = subject_labels.shape
        if F_lab % 3 == 0:
            num_markers = F_lab // 3
            # Assume interleaved layout (x,y,z,x,y,z,...)
            markers_xyz = subject_labels.reshape(N_lab, T_lab, num_markers, 3)
            print(f"[info] {num_markers} markers loaded")
        else:
            print(f"[warn] Marker data has {F_lab} features (not divisible by 3), skipping")
            markers_xyz = None
    else:
        markers_xyz = None
        num_markers = 0

    # Reconstruct continuous motion (skeleton)
    continuous_motion = reconstruct_continuous_motion(sequences, stride=args.stride)

    # Reconstruct continuous markers if available
    if markers_xyz is not None:
        continuous_markers = reconstruct_continuous_motion(markers_xyz, stride=args.stride)
    else:
        continuous_markers = None

    # Rotate keypoints +90° around X axis to make Y point up instead of Z
    # Rotation: (x, y, z) -> (x, -z, y)
    # This is R_x(+90°) = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    rotated_motion = continuous_motion.copy()
    rotated_motion[..., 0] = continuous_motion[..., 0]   # x stays
    rotated_motion[..., 1] = -continuous_motion[..., 2]  # y = -z
    rotated_motion[..., 2] = continuous_motion[..., 1]   # z = y
    continuous_motion = rotated_motion

    # Create centroidal dynamics calculator
    dt = 1.0 / args.fps
    dynamics = CentroidalDynamics(total_mass=weight, height=height, dt=dt)

    print(f"[info] Total body mass: {weight:.2f} kg")

    # Setup visualization
    vis = meshcat.Visualizer().open()
    marker_count = continuous_markers.shape[1] if continuous_markers is not None else 0
    setup_scene_with_dynamics(vis, args.radius, K, num_markers=marker_count)
    edges = EDGES if K == 20 else [(i, i+1) for i in range(K-1)]


    print(f"[info] Layout: {layout}")
    print("[info] Visualization:")
    print("  - White spheres: skeleton joints")
    print("  - Red sphere: center of mass")
    print("  - Blue arrow: linear momentum (from midHip)")
    print("  - Green disc+arrow: angular momentum (about COM)")
    print("\n[info] Press Ctrl+C to stop\n")

    # Play motion with dynamics
    try:
        while True:
            play_continuous_motion_with_dynamics(
                vis, continuous_motion, dynamics, args.fps, edges,
                momentum_scale=args.momentum_scale,
                angular_momentum_scale=args.angular_momentum_scale,
                markers=continuous_markers
            )
            if not args.loop:
                break
            print("[info] Looping...")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[info] Stopped.")


if __name__ == "__main__":
    main()

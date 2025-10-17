#!/usr/bin/env python3
"""
Example script demonstrating centroidal dynamics computation on real dataset.

Usage:
    python example_centroidal_dynamics.py --dataset-root ./datasets/dataset0 --subject 3
"""
import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from centroidal_dynamics import (
    CentroidalDynamics,
    extract_height_weight_from_features,
    create_dynamics_from_features
)


def load_dataset(dataset_root: Path):
    """Load dataset features and subjects."""
    info = np.load(dataset_root / "infoData.npy", allow_pickle=True)
    if isinstance(info, np.ndarray) and info.dtype == object:
        info = info.item()
    subjects = np.asarray(info["subjects"])

    with h5py.File(dataset_root / "time_sequences.h5", "r") as f:
        features = f["data/features"][:]

    return features, subjects


def get_subject_sequences(features, subjects, subject_id):
    """Get all sequences for a specific subject."""
    mask = subjects == subject_id
    return features[mask]


def features_to_xyz(features: np.ndarray) -> np.ndarray:
    """Convert features to XYZ joint positions (assuming interleaved layout)."""
    # Strip height/weight (last 2 columns)
    if features.shape[-1] == 62:
        joint_data = features[:, :, :-2]  # (N, T, 60)
    else:
        joint_data = features

    N, T, F = joint_data.shape
    K = F // 3
    return joint_data.reshape(N, T, K, 3)


def main():
    parser = argparse.ArgumentParser(description="Compute and visualize centroidal dynamics")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--seq-idx", type=int, default=0, help="Sequence index to analyze")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--plot", action="store_true", help="Show plots")

    args = parser.parse_args()

    # Load data
    print("Loading dataset...")
    features, subjects = load_dataset(Path(args.dataset_root))
    subject_features = get_subject_sequences(features, subjects, args.subject)

    print(f"Subject {args.subject}: {len(subject_features)} sequences")

    # Get one sequence
    seq_features = subject_features[args.seq_idx]  # (T, 62)
    print(f"Analyzing sequence {args.seq_idx}: shape={seq_features.shape}")

    # Extract height and weight
    height, weight = extract_height_weight_from_features(seq_features)
    print(f"\nSubject parameters:")
    print(f"  Height: {height:.3f} m")
    print(f"  Weight: {weight:.3f} kg")

    # Convert to XYZ joints
    xyz_joints = features_to_xyz(seq_features[np.newaxis, ...])[0]  # (T, K, 3)
    T, K, _ = xyz_joints.shape
    print(f"\nJoint data: {T} frames, {K} joints")

    # Create dynamics calculator
    dt = 1.0 / args.fps
    dynamics = CentroidalDynamics(total_mass=weight, height=height, dt=dt)

    # Show segment information
    print("\n" + "="*60)
    print("Body Segment Masses")
    print("="*60)
    segment_info = dynamics.get_segment_info()
    for seg_name, info in segment_info.items():
        print(f"{seg_name:20s}: {info['mass']:5.2f} kg ({info['mass_fraction']*100:4.1f}%)")

    # Compute all dynamics
    print("\n" + "="*60)
    print("Computing Centroidal Dynamics")
    print("="*60)
    result = dynamics.compute_all_dynamics(xyz_joints)

    # Print statistics
    com = result['com']
    com_vel = result['com_velocity']
    lin_mom = result['linear_momentum']
    ang_mom = result['angular_momentum']

    print(f"\nCenter of Mass:")
    print(f"  Mean position: [{com.mean(axis=0)[0]:.3f}, {com.mean(axis=0)[1]:.3f}, {com.mean(axis=0)[2]:.3f}] m")
    print(f"  Range: X=[{com[:,0].min():.3f}, {com[:,0].max():.3f}] m")
    print(f"         Y=[{com[:,1].min():.3f}, {com[:,1].max():.3f}] m")
    print(f"         Z=[{com[:,2].min():.3f}, {com[:,2].max():.3f}] m")

    print(f"\nCOM Velocity:")
    speed = np.linalg.norm(com_vel, axis=1)
    print(f"  Mean speed: {speed.mean():.3f} m/s")
    print(f"  Max speed: {speed.max():.3f} m/s")

    print(f"\nLinear Momentum:")
    lin_mom_mag = np.linalg.norm(lin_mom, axis=1)
    print(f"  Mean magnitude: {lin_mom_mag.mean():.3f} kg⋅m/s")
    print(f"  Max magnitude: {lin_mom_mag.max():.3f} kg⋅m/s")

    print(f"\nAngular Momentum (about COM):")
    ang_mom_mag = np.linalg.norm(ang_mom, axis=1)
    print(f"  Mean magnitude: {ang_mom_mag.mean():.3f} kg⋅m²/s")
    print(f"  Max magnitude: {ang_mom_mag.max():.3f} kg⋅m²/s")

    # Plotting
    if args.plot:
        print("\nGenerating plots...")
        fig = plt.figure(figsize=(15, 10))

        # 1. COM trajectory in 3D
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(com[:, 0], com[:, 1], com[:, 2], 'b-', linewidth=2)
        ax1.scatter(com[0, 0], com[0, 1], com[0, 2], c='g', s=100, marker='o', label='Start')
        ax1.scatter(com[-1, 0], com[-1, 1], com[-1, 2], c='r', s=100, marker='x', label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('COM Trajectory')
        ax1.legend()

        # 2. COM position over time
        ax2 = fig.add_subplot(2, 3, 2)
        time = np.arange(T) * dt
        ax2.plot(time, com[:, 0], label='X', linewidth=2)
        ax2.plot(time, com[:, 1], label='Y', linewidth=2)
        ax2.plot(time, com[:, 2], label='Z', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('COM Position vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. COM velocity
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time, com_vel[:, 0], label='Vx', linewidth=2)
        ax3.plot(time, com_vel[:, 1], label='Vy', linewidth=2)
        ax3.plot(time, com_vel[:, 2], label='Vz', linewidth=2)
        ax3.plot(time, speed, 'k--', label='Speed', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('COM Velocity vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Linear momentum
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time, lin_mom[:, 0], label='Px', linewidth=2)
        ax4.plot(time, lin_mom[:, 1], label='Py', linewidth=2)
        ax4.plot(time, lin_mom[:, 2], label='Pz', linewidth=2)
        ax4.plot(time, lin_mom_mag, 'k--', label='Magnitude', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Momentum (kg⋅m/s)')
        ax4.set_title('Linear Momentum vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Angular momentum
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(time, ang_mom[:, 0], label='Lx', linewidth=2)
        ax5.plot(time, ang_mom[:, 1], label='Ly', linewidth=2)
        ax5.plot(time, ang_mom[:, 2], label='Lz', linewidth=2)
        ax5.plot(time, ang_mom_mag, 'k--', label='Magnitude', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angular Momentum (kg⋅m²/s)')
        ax5.set_title('Angular Momentum vs Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Segment mass distribution
        ax6 = fig.add_subplot(2, 3, 6)
        seg_names = list(segment_info.keys())
        seg_masses = [segment_info[name]['mass'] for name in seg_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(seg_names)))
        bars = ax6.barh(seg_names, seg_masses, color=colors)
        ax6.set_xlabel('Mass (kg)')
        ax6.set_title('Segment Mass Distribution')
        ax6.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

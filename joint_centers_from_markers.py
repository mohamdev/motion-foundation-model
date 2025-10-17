#!/usr/bin/env python3
"""
Compute anatomical joint centers from motion capture markers.

This module derives joint center positions from marker locations using
anatomical definitions and geometric calculations. This provides more
accurate joint positions than using the raw feature joint data.

Marker indices (from the 43-marker set):
    0-39: Various anatomical markers
    41: RHJC_augmenter (Right Hip Joint Center)
    42: LHJC_augmenter (Left Hip Joint Center)
"""
import numpy as np
from typing import Dict, Tuple, Optional


# Marker name to index mapping
# Note: The last 2 markers (RHJC_gt, LHJC_gt) are ground truth hip joint centers
MARKER_NAMES = [
    "RFHD", "LFHD", "RBHD", "LBHD", "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
    "LMWRIST", "RMWRIST", "LMELBOW", "RMELBOW",
    "RHJC_gt", "LHJC_gt"
]

MARKER_IDX = {name: i for i, name in enumerate(MARKER_NAMES)}


def midpoint(markers: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
    """Compute midpoint between two markers."""
    return (markers[:, idx1] + markers[:, idx2]) / 2.0


def compute_wrist_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute wrist joint center as barycenter of wrist markers.

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        wrist_center: (T, 3) wrist joint center positions
    """
    prefix = 'R' if side == 'R' else 'L'
    wra_idx = MARKER_IDX[f"{prefix}WRA"]  # Radial styloid
    wrb_idx = MARKER_IDX[f"{prefix}WRB"]  # Ulnar styloid
    mwrist_idx = MARKER_IDX[f"{prefix}MWRIST"]  # Medial wrist marker

    # Wrist center is barycenter of all wrist markers
    return (markers[:, wra_idx] + markers[:, wrb_idx] + markers[:, mwrist_idx]) / 3.0


def compute_elbow_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute elbow joint center as barycenter of elbow markers.

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        elbow_center: (T, 3) elbow joint center positions
    """
    prefix = 'R' if side == 'R' else 'L'

    melb_idx = MARKER_IDX[f"{prefix}MELBOW"]  # Medial epicondyle
    elb_idx = MARKER_IDX[f"{prefix}ELB"]      # Lateral epicondyle

    # Elbow center is barycenter (midpoint in this case - 2 markers)
    return (markers[:, melb_idx] + markers[:, elb_idx]) / 2.0


def compute_knee_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute knee joint center as barycenter of knee markers.

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        knee_center: (T, 3) knee joint center positions
    """
    prefix = 'R' if side == 'R' else 'L'
    kne_idx = MARKER_IDX[f"{prefix}KNE"]      # Lateral knee marker
    thi_idx = MARKER_IDX[f"{prefix}THI"]      # Thigh marker (for context)

    # Use knee marker as center (only one knee marker available)
    # In future could compute barycenter if medial knee marker exists
    return markers[:, kne_idx]


def compute_ankle_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute ankle joint center as barycenter of ankle markers.

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        ankle_center: (T, 3) ankle joint center positions
    """
    prefix = 'R' if side == 'R' else 'L'
    ank_idx = MARKER_IDX[f"{prefix}ANK"]      # Lateral malleolus
    tib_idx = MARKER_IDX[f"{prefix}TIB"]      # Tibia marker

    # Ankle center is barycenter of ankle markers
    return (markers[:, ank_idx] + markers[:, tib_idx]) / 2.0


def compute_heel_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute heel center from heel marker (RHEE/LHEE).

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        heel_center: (T, 3) heel positions
    """
    prefix = 'R' if side == 'R' else 'L'
    hee_idx = MARKER_IDX[f"{prefix}HEE"]
    return markers[:, hee_idx]


def compute_toe_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Compute toe center from toe marker (RTOE/LTOE).

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        toe_center: (T, 3) toe positions
    """
    prefix = 'R' if side == 'R' else 'L'
    toe_idx = MARKER_IDX[f"{prefix}TOE"]
    return markers[:, toe_idx]


def compute_hip_center(markers: np.ndarray, side: str) -> np.ndarray:
    """
    Use ground truth hip joint center (RHJC_gt/LHJC_gt).

    Args:
        markers: (T, M, 3) marker positions
        side: 'R' or 'L' for right/left

    Returns:
        hip_center: (T, 3) hip joint center positions
    """
    prefix = 'R' if side == 'R' else 'L'
    hjc_idx = MARKER_IDX[f"{prefix}HJC_gt"]
    return markers[:, hjc_idx]


def compute_midhip_from_markers(markers: np.ndarray) -> np.ndarray:
    """
    Compute midHip as barycenter of hip joint centers.

    Args:
        markers: (T, M, 3) marker positions

    Returns:
        midhip: (T, 3) midHip positions
    """
    rhjc = compute_hip_center(markers, 'R')
    lhjc = compute_hip_center(markers, 'L')
    return (rhjc + lhjc) / 2.0


def compute_all_joint_centers(markers: np.ndarray,
                              feature_joints: np.ndarray) -> np.ndarray:
    """
    Compute all 20 joint centers from markers.

    Joint order (matching JOINT_NAMES):
    [Neck, RShoulder, LShoulder, RHip, LHip, midHip,
     RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel,
     RSmallToe, LSmallToe, RBigToe, LBigToe,
     RElbow, LElbow, RWrist, LWrist]

    Strategy:
    - Neck, Shoulders: keep from feature_joints
    - Hips: use RHJC_gt/LHJC_gt, compute midHip as barycenter
    - Elbows, Wrists: compute as barycenter of corresponding markers
    - Knees, Ankles: compute as barycenter of corresponding markers
    - Heels, Toes: use RHEE/LHEE and RTOE/LTOE markers

    Args:
        markers: (T, M, 3) marker positions
        feature_joints: (T, 20, 3) original joint positions from features

    Returns:
        joints: (T, 20, 3) joint center positions
    """
    T = markers.shape[0]
    joints = np.zeros((T, 20, 3), dtype=markers.dtype)

    # Joint indices
    Neck, RSh, LSh, RHip, LHip, midHip = 0, 1, 2, 3, 4, 5
    RKnee, LKnee, RAnk, LAnk, RHeel, LHeel = 6, 7, 8, 9, 10, 11
    RST, LST, RBT, LBT = 12, 13, 14, 15
    RElb, LElb, RWri, LWri = 16, 17, 18, 19

    # Keep neck and shoulders from features
    joints[:, Neck] = feature_joints[:, Neck]
    joints[:, RSh] = feature_joints[:, RSh]
    joints[:, LSh] = feature_joints[:, LSh]

    # Hips: use ground truth markers
    joints[:, RHip] = compute_hip_center(markers, 'R')
    joints[:, LHip] = compute_hip_center(markers, 'L')
    joints[:, midHip] = compute_midhip_from_markers(markers)

    # Knees: barycenter of knee markers
    joints[:, RKnee] = compute_knee_center(markers, 'R')
    joints[:, LKnee] = compute_knee_center(markers, 'L')

    # Ankles: barycenter of ankle markers
    joints[:, RAnk] = compute_ankle_center(markers, 'R')
    joints[:, LAnk] = compute_ankle_center(markers, 'L')

    # Heels: use heel markers
    joints[:, RHeel] = compute_heel_center(markers, 'R')
    joints[:, LHeel] = compute_heel_center(markers, 'L')

    # Toes: use toe markers (same for small/big toe)
    joints[:, RST] = compute_toe_center(markers, 'R')
    joints[:, LST] = compute_toe_center(markers, 'L')
    joints[:, RBT] = compute_toe_center(markers, 'R')
    joints[:, LBT] = compute_toe_center(markers, 'L')

    # Elbows: barycenter of elbow markers
    joints[:, RElb] = compute_elbow_center(markers, 'R')
    joints[:, LElb] = compute_elbow_center(markers, 'L')

    # Wrists: barycenter of wrist markers
    joints[:, RWri] = compute_wrist_center(markers, 'R')
    joints[:, LWri] = compute_wrist_center(markers, 'L')

    return joints


# ===============================================================
# Example usage
# ===============================================================

if __name__ == "__main__":
    print("Joint Centers from Markers Module")
    print("=" * 60)

    # Example with synthetic marker data
    T = 100  # frames
    M = 43   # markers

    np.random.seed(42)
    markers = np.random.randn(T, M, 3) * 0.1

    # Compute joint centers
    joints = compute_all_joint_centers(markers, keep_shoulder_from_features=False)

    print(f"\nComputed joint centers from {M} markers")
    print(f"Output shape: {joints.shape} (T={T}, K=20, 3)")
    print(f"\nJoint center positions at frame 0:")

    joint_names = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "midHip",
        "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
        "RSmallToe", "LSmallToe", "RBigToe", "LBigToe",
        "RElbow", "LElbow", "RWrist", "LWrist"
    ]

    for i, name in enumerate(joint_names):
        pos = joints[0, i]
        print(f"  {name:15s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

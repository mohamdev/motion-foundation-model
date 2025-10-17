#!/usr/bin/env python3
"""
Centroidal dynamics computation for 3D skeleton motion.

This module provides tools to compute center of mass, linear momentum,
and angular momentum from skeletal joint positions using anthropometric
body segment parameters.

References:
    - Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement. 4th ed.
    - de Leva, P. (1996). Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters.
      Journal of Biomechanics, 29(9), 1223-1230.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SegmentParameters:
    """Parameters for a body segment."""
    name: str
    mass_fraction: float  # Fraction of total body mass
    com_ratio: float      # COM position ratio from proximal joint
    radius_of_gyration: float  # Radius of gyration as fraction of segment length


class AnthropometricModel:
    """
    Anthropometric model for body segment parameters.

    Uses de Leva (1996) adjustments to Zatsiorsky-Seluyanov parameters,
    which are more recent and accurate than Dempster or Winter tables.
    """

    # de Leva (1996) parameters - averaged male/female values
    SEGMENT_PARAMS = {
        # Trunk and head
        "head": SegmentParameters("head", 0.0694, 0.5, 0.303),
        "trunk": SegmentParameters("trunk", 0.4346, 0.5, 0.372),

        # Upper extremities
        "upper_arm": SegmentParameters("upper_arm", 0.0271, 0.436, 0.322),
        "forearm": SegmentParameters("forearm", 0.0162, 0.430, 0.303),
        "hand": SegmentParameters("hand", 0.0061, 0.506, 0.297),

        # Lower extremities
        "thigh": SegmentParameters("thigh", 0.1416, 0.433, 0.329),
        "shank": SegmentParameters("shank", 0.0433, 0.433, 0.302),
        "foot": SegmentParameters("foot", 0.0137, 0.500, 0.257),
    }

    @classmethod
    def get_segment_mass(cls, segment_name: str, total_mass: float) -> float:
        """Get mass of a body segment."""
        params = cls.SEGMENT_PARAMS.get(segment_name)
        if params is None:
            raise ValueError(f"Unknown segment: {segment_name}")
        return params.mass_fraction * total_mass

    @classmethod
    def get_segment_com_ratio(cls, segment_name: str) -> float:
        """Get COM position ratio from proximal joint."""
        params = cls.SEGMENT_PARAMS.get(segment_name)
        if params is None:
            raise ValueError(f"Unknown segment: {segment_name}")
        return params.com_ratio


class BodySegmentDefinition:
    """
    Maps skeleton joints to body segments.

    Assumes the 20-joint skeleton structure:
    [Neck, RShoulder, LShoulder, RHip, LHip, midHip,
     RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel,
     RSmallToe, LSmallToe, RBigToe, LBigToe,
     RElbow, LElbow, RWrist, LWrist]
    """

    # Joint indices
    Neck, RSh, LSh, RHip, LHip, midHip = 0, 1, 2, 3, 4, 5
    RKnee, LKnee, RAnk, LAnk, RHeel, LHeel = 6, 7, 8, 9, 10, 11
    RST, LST, RBT, LBT = 12, 13, 14, 15
    RElb, LElb, RWri, LWri = 16, 17, 18, 19

    # Segment definitions: (proximal_joint, distal_joint, segment_type, bilateral)
    # For bilateral segments, we'll create separate left/right instances
    SEGMENTS = {
        # Head and trunk (using Neck as head top, midHip as pelvis)
        "head": (Neck, Neck, "head", False),  # Point mass at neck
        "trunk": (Neck, midHip, "trunk", False),

        # Right upper extremity
        "right_upper_arm": (RSh, RElb, "upper_arm", True),
        "right_forearm": (RElb, RWri, "forearm", True),
        "right_hand": (RWri, RWri, "hand", True),  # Point mass at wrist

        # Left upper extremity
        "left_upper_arm": (LSh, LElb, "upper_arm", True),
        "left_forearm": (LElb, LWri, "forearm", True),
        "left_hand": (LWri, LWri, "hand", True),

        # Right lower extremity
        "right_thigh": (RHip, RKnee, "thigh", True),
        "right_shank": (RKnee, RAnk, "shank", True),
        "right_foot": (RAnk, RBT, "foot", True),  # Ankle to big toe

        # Left lower extremity
        "left_thigh": (LHip, LKnee, "thigh", True),
        "left_shank": (LKnee, LAnk, "shank", True),
        "left_foot": (LAnk, LBT, "foot", True),
    }


class CentroidalDynamics:
    """
    Compute centroidal dynamics from skeletal motion data.

    This class calculates:
    - Center of mass (COM) position
    - Linear momentum
    - Angular momentum (about COM)

    Attributes:
        total_mass: Total body mass in kg
        height: Body height in meters
        dt: Time step between frames (for velocity/momentum calculation)
        anthropometric_model: Anthropometric model to use
    """

    def __init__(self, total_mass: float, height: float, dt: float = 1/30.0,
                 anthropometric_model: type = AnthropometricModel):
        """
        Initialize centroidal dynamics calculator.

        Args:
            total_mass: Total body mass in kg
            height: Body height in meters
            dt: Time step between frames in seconds (default: 1/30 for 30 fps)
            anthropometric_model: Anthropometric model class to use
        """
        self.total_mass = total_mass
        self.height = height
        self.dt = dt
        self.anthropometric_model = anthropometric_model

        # Precompute segment masses
        self.segment_masses = self._compute_segment_masses()

    def _compute_segment_masses(self) -> Dict[str, float]:
        """Precompute mass for each body segment."""
        segment_masses = {}
        for seg_name, (prox, dist, seg_type, bilateral) in BodySegmentDefinition.SEGMENTS.items():
            mass = self.anthropometric_model.get_segment_mass(seg_type, self.total_mass)
            segment_masses[seg_name] = mass
        return segment_masses

    def compute_segment_com(self, joints: np.ndarray, segment_name: str) -> np.ndarray:
        """
        Compute center of mass position for a body segment.

        Args:
            joints: (K, 3) array of joint positions for one frame
            segment_name: Name of the segment

        Returns:
            com: (3,) array of segment COM position
        """
        prox_idx, dist_idx, seg_type, _ = BodySegmentDefinition.SEGMENTS[segment_name]
        prox_pos = joints[prox_idx]
        dist_pos = joints[dist_idx]

        # Get COM ratio from proximal joint
        com_ratio = self.anthropometric_model.get_segment_com_ratio(seg_type)

        # Compute COM position
        com = prox_pos + com_ratio * (dist_pos - prox_pos)
        return com

    def compute_center_of_mass(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute whole-body center of mass position.

        Args:
            joints: (T, K, 3) or (K, 3) array of joint positions
                   T = time frames, K = number of joints, 3 = xyz

        Returns:
            com: (T, 3) or (3,) array of COM positions
        """
        single_frame = joints.ndim == 2
        if single_frame:
            joints = joints[np.newaxis, ...]  # Add time dimension

        T, K, _ = joints.shape
        com = np.zeros((T, 3))

        for t in range(T):
            weighted_sum = np.zeros(3)
            for seg_name, mass in self.segment_masses.items():
                seg_com = self.compute_segment_com(joints[t], seg_name)
                weighted_sum += mass * seg_com
            com[t] = weighted_sum / self.total_mass

        return com[0] if single_frame else com

    def compute_segment_velocity(self, joints: np.ndarray, segment_name: str) -> np.ndarray:
        """
        Compute center of mass velocity for a body segment.

        Args:
            joints: (T, K, 3) array of joint positions
            segment_name: Name of the segment

        Returns:
            velocity: (T, 3) array of segment COM velocities
        """
        T = joints.shape[0]
        segment_coms = np.zeros((T, 3))

        for t in range(T):
            segment_coms[t] = self.compute_segment_com(joints[t], segment_name)

        # Compute velocity using central differences
        velocity = np.zeros_like(segment_coms)
        velocity[1:-1] = (segment_coms[2:] - segment_coms[:-2]) / (2 * self.dt)
        # Forward/backward differences at boundaries
        velocity[0] = (segment_coms[1] - segment_coms[0]) / self.dt
        velocity[-1] = (segment_coms[-1] - segment_coms[-2]) / self.dt

        return velocity

    def compute_linear_momentum(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute whole-body linear momentum.

        Args:
            joints: (T, K, 3) array of joint positions

        Returns:
            momentum: (T, 3) array of linear momentum vectors [kg⋅m/s]
        """
        T = joints.shape[0]
        momentum = np.zeros((T, 3))

        for seg_name, mass in self.segment_masses.items():
            seg_velocity = self.compute_segment_velocity(joints, seg_name)
            momentum += mass * seg_velocity

        return momentum

    def compute_angular_momentum(self, joints: np.ndarray,
                                about_point: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute whole-body angular momentum.

        Args:
            joints: (T, K, 3) array of joint positions
            about_point: (T, 3) or (3,) array of point about which to compute
                        angular momentum. If None, uses whole-body COM.

        Returns:
            angular_momentum: (T, 3) array of angular momentum vectors [kg⋅m²/s]
        """
        T = joints.shape[0]

        # Compute reference point (default: COM)
        if about_point is None:
            about_point = self.compute_center_of_mass(joints)
        elif about_point.ndim == 1:
            about_point = np.tile(about_point, (T, 1))

        angular_momentum = np.zeros((T, 3))

        for seg_name, mass in self.segment_masses.items():
            # Get segment COM position and velocity
            seg_com = np.zeros((T, 3))
            for t in range(T):
                seg_com[t] = self.compute_segment_com(joints[t], seg_name)

            seg_velocity = self.compute_segment_velocity(joints, seg_name)

            # Compute angular momentum: L = r × (m * v)
            for t in range(T):
                r = seg_com[t] - about_point[t]
                momentum = mass * seg_velocity[t]
                angular_momentum[t] += np.cross(r, momentum)

        return angular_momentum

    def compute_com_velocity(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute center of mass velocity.

        Args:
            joints: (T, K, 3) array of joint positions

        Returns:
            com_velocity: (T, 3) array of COM velocities [m/s]
        """
        com = self.compute_center_of_mass(joints)

        # Compute velocity using central differences
        T = com.shape[0]
        velocity = np.zeros_like(com)
        velocity[1:-1] = (com[2:] - com[:-2]) / (2 * self.dt)
        velocity[0] = (com[1] - com[0]) / self.dt
        velocity[-1] = (com[-1] - com[-2]) / self.dt

        return velocity

    def compute_all_dynamics(self, joints: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all centroidal dynamics quantities.

        Args:
            joints: (T, K, 3) array of joint positions

        Returns:
            dynamics: Dictionary containing:
                - 'com': (T, 3) center of mass positions [m]
                - 'com_velocity': (T, 3) COM velocities [m/s]
                - 'linear_momentum': (T, 3) linear momentum [kg⋅m/s]
                - 'angular_momentum': (T, 3) angular momentum about COM [kg⋅m²/s]
        """
        com = self.compute_center_of_mass(joints)
        com_velocity = self.compute_com_velocity(joints)
        linear_momentum = self.compute_linear_momentum(joints)
        angular_momentum = self.compute_angular_momentum(joints)

        return {
            'com': com,
            'com_velocity': com_velocity,
            'linear_momentum': linear_momentum,
            'angular_momentum': angular_momentum,
        }

    def get_segment_info(self) -> Dict[str, Dict]:
        """
        Get information about all body segments.

        Returns:
            segment_info: Dictionary mapping segment names to their properties
        """
        info = {}
        for seg_name, mass in self.segment_masses.items():
            prox_idx, dist_idx, seg_type, bilateral = BodySegmentDefinition.SEGMENTS[seg_name]
            info[seg_name] = {
                'mass': mass,
                'mass_fraction': mass / self.total_mass,
                'segment_type': seg_type,
                'proximal_joint': prox_idx,
                'distal_joint': dist_idx,
                'bilateral': bilateral,
            }
        return info


# ===============================================================
# Utility functions
# ===============================================================

def extract_height_weight_from_features(features: np.ndarray) -> Tuple[float, float]:
    """
    Extract height and weight from dataset features (F=62).

    Args:
        features: (T, 62) or (N, T, 62) array with last 2 columns = [height, weight]

    Returns:
        height: Height in meters
        weight: Weight in kilograms
    """
    if features.shape[-1] != 62:
        raise ValueError(f"Expected 62 features, got {features.shape[-1]}")

    # Height and weight are constant across frames
    if features.ndim == 3:
        features = features[0]  # Take first sequence

    height = float(features[0, -2])
    weight = float(features[0, -1])

    return height, weight


def create_dynamics_from_features(features: np.ndarray, dt: float = 1/30.0) -> CentroidalDynamics:
    """
    Create CentroidalDynamics instance from dataset features.

    Args:
        features: (T, 62) or (N, T, 62) array with last 2 columns = [height, weight]
        dt: Time step in seconds

    Returns:
        dynamics: CentroidalDynamics instance
    """
    height, weight = extract_height_weight_from_features(features)
    return CentroidalDynamics(total_mass=weight, height=height, dt=dt)


# ===============================================================
# Example usage
# ===============================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("Centroidal Dynamics Module")
    print("=" * 60)

    # Create example skeleton motion (10 frames, 20 joints)
    T, K = 10, 20
    np.random.seed(42)
    joints = np.random.randn(T, K, 3) * 0.1

    # Example subject parameters
    height = 1.75  # meters
    weight = 70.0  # kg
    dt = 1/30.0    # 30 fps

    # Create dynamics calculator
    dynamics = CentroidalDynamics(total_mass=weight, height=height, dt=dt)

    # Compute all dynamics
    result = dynamics.compute_all_dynamics(joints)

    print(f"\nSubject: height={height}m, weight={weight}kg")
    print(f"Time step: {dt}s ({1/dt:.0f} fps)")
    print(f"\nComputed dynamics for {T} frames:")
    print(f"  - Center of mass: {result['com'].shape}")
    print(f"  - COM velocity: {result['com_velocity'].shape}")
    print(f"  - Linear momentum: {result['linear_momentum'].shape}")
    print(f"  - Angular momentum: {result['angular_momentum'].shape}")

    # Show segment information
    print("\nBody segment masses:")
    for seg_name, info in dynamics.get_segment_info().items():
        print(f"  {seg_name:20s}: {info['mass']:5.2f} kg ({info['mass_fraction']*100:4.1f}%)")

    print(f"\nTotal mass check: {sum(dynamics.segment_masses.values()):.2f} kg")

"""
Enhanced script to explore pose dataset structure and visualize 3D poses
with detailed analysis of keypoints/joints per frame.
"""

import numpy as np
import h5py
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

def analyze_metadata_structure(info_data):
    """Analyze the metadata structure to understand dataset organization"""
    print("\n" + "="*60)
    print("METADATA STRUCTURE ANALYSIS")
    print("="*60)
    
    if isinstance(info_data, dict):
        # Analyze subjects
        if 'subjects' in info_data:
            subjects = info_data['subjects']
            unique_subjects = np.unique(subjects)
            print(f"\nSubject Information:")
            print(f"  Total unique subjects: {len(unique_subjects)}")
            print(f"  Subject IDs: {unique_subjects}")
            
            # Count frames per subject
            subject_counts = Counter(subjects)
            print(f"\n  Frames per subject:")
            for subj_id in sorted(subject_counts.keys()):
                print(f"    Subject {subj_id}: {subject_counts[subj_id]} frames")
        
        # Analyze scale factors
        if 'scalefactors' in info_data:
            scale_factors = info_data['scalefactors']
            unique_scales = np.unique(scale_factors)
            print(f"\nScale Factor Information:")
            print(f"  Unique scale factors: {unique_scales}")
            
            # Count frames per scale factor
            scale_counts = Counter(scale_factors)
            print(f"\n  Frames per scale factor:")
            for scale in sorted(scale_counts.keys()):
                print(f"    Scale {scale}: {scale_counts[scale]} frames")
        
        # Cross-analysis: subjects vs scale factors
        if 'subjects' in info_data and 'scalefactors' in info_data:
            print(f"\nCross-analysis (Subject vs Scale Factor):")
            for subj in unique_subjects:
                subj_mask = subjects == subj
                subj_scales = scale_factors[subj_mask]
                unique_subj_scales = np.unique(subj_scales)
                print(f"  Subject {subj}: scales {unique_subj_scales}, "
                      f"total frames: {len(subj_scales)}")

def explore_h5_detailed(filepath):
    """Detailed exploration of HDF5 file with focus on pose structure"""
    print("\n" + "="*60)
    print("DETAILED H5 POSE DATA EXPLORATION")
    print("="*60)
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"\nHDF5 file contents:")
            
            # First, get all datasets recursively
            all_datasets = {}
            
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets[name] = {
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'size': obj.size
                    }
            
            f.visititems(find_datasets)
            
            print(f"\nFound {len(all_datasets)} datasets:")
            for name, info in all_datasets.items():
                print(f"\n  Dataset: '{name}'")
                print(f"    Shape: {info['shape']}")
                print(f"    Dtype: {info['dtype']}")
                print(f"    Total elements: {info['size']}")
                
                # Analyze if this could be pose data
                shape = info['shape']
                if len(shape) >= 2:
                    last_dim = shape[-1]
                    
                    # Check if last dimension is divisible by 3 (for x,y,z coordinates)
                    if last_dim % 3 == 0:
                        num_keypoints = last_dim // 3
                        print(f"    → Potential 3D pose data with {num_keypoints} keypoints")
                    
                    # Check if last dimension is divisible by 2 (for x,y coordinates)
                    elif last_dim % 2 == 0:
                        num_keypoints = last_dim // 2
                        print(f"    → Potential 2D pose data with {num_keypoints} keypoints")
                    
                    # Analyze data dimensions
                    if len(shape) == 3:
                        print(f"    → 3D array: {shape[0]} samples × {shape[1]} timesteps × {shape[2]} features")
                    elif len(shape) == 2:
                        print(f"    → 2D array: {shape[0]} samples × {shape[1]} features")
            
            # Load and analyze actual data for the largest dataset
            if all_datasets:
                # Find the dataset with most data
                largest_dataset = max(all_datasets.items(), 
                                    key=lambda x: x[1]['size'])
                dataset_name = largest_dataset[0]
                
                print(f"\n" + "-"*40)
                print(f"Analyzing largest dataset: '{dataset_name}'")
                print("-"*40)
                
                data = f[dataset_name][:]
                analyze_pose_data(data, dataset_name)
                
                return data, dataset_name
            
    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        return None, None

def analyze_pose_data(data, dataset_name):
    """Analyze pose data structure and statistics"""
    print(f"\nPose Data Analysis for '{dataset_name}':")
    print(f"  Data shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    
    # Determine structure
    if len(data.shape) == 3:
        n_samples, n_timesteps, n_features = data.shape
        print(f"\n  Structure: {n_samples} sequences")
        print(f"    Each sequence: {n_timesteps} timesteps")
        print(f"    Each timestep: {n_features} features")
        
        # Analyze feature dimension
        if n_features % 3 == 0:
            n_keypoints = n_features // 3
            print(f"\n  Detected 3D poses:")
            print(f"    Number of keypoints/joints: {n_keypoints}")
            print(f"    Coordinates per keypoint: 3 (x, y, z)")
        elif n_features % 2 == 0:
            n_keypoints = n_features // 2
            print(f"\n  Detected 2D poses:")
            print(f"    Number of keypoints/joints: {n_keypoints}")
            print(f"    Coordinates per keypoint: 2 (x, y)")
        
        # Statistics
        print(f"\n  Data Statistics:")
        print(f"    Global min: {np.min(data):.4f}")
        print(f"    Global max: {np.max(data):.4f}")
        print(f"    Global mean: {np.mean(data):.4f}")
        print(f"    Global std: {np.std(data):.4f}")
        
        # Per-dimension statistics (assuming 3D)
        if n_features % 3 == 0:
            n_keypoints = n_features // 3
            print(f"\n  Per-coordinate statistics:")
            for dim, name in enumerate(['X', 'Y', 'Z']):
                dim_data = data[:, :, dim::3]  # Every 3rd element starting from dim
                print(f"    {name}-coordinate: "
                      f"min={np.min(dim_data):.4f}, "
                      f"max={np.max(dim_data):.4f}, "
                      f"mean={np.mean(dim_data):.4f}, "
                      f"std={np.std(dim_data):.4f}")
        
        # Check for missing/invalid data
        print(f"\n  Data Quality:")
        print(f"    Contains NaN: {np.any(np.isnan(data))}")
        print(f"    Contains Inf: {np.any(np.isinf(data))}")
        print(f"    Zero values: {np.sum(data == 0)} ({100 * np.sum(data == 0) / data.size:.2f}%)")
        
    elif len(data.shape) == 2:
        n_samples, n_features = data.shape
        print(f"\n  Structure: {n_samples} frames")
        print(f"    Each frame: {n_features} features")
        
        if n_features % 3 == 0:
            n_keypoints = n_features // 3
            print(f"\n  Detected 3D poses:")
            print(f"    Number of keypoints/joints: {n_keypoints}")

def visualize_pose_sequence(data, sample_idx=0, timesteps_to_show=5):
    """Visualize multiple timesteps from a pose sequence"""
    print("\n" + "="*60)
    print("POSE SEQUENCE VISUALIZATION")
    print("="*60)
    
    if data is None:
        print("No data available for visualization")
        return
    
    # Handle different data shapes
    if len(data.shape) == 3:
        # Shape: [samples, timesteps, features]
        sequence = data[sample_idx]  # Get one sequence
        n_timesteps, n_features = sequence.shape
        
        if n_features % 3 == 0:
            n_keypoints = n_features // 3
            print(f"\nVisualizing sequence {sample_idx}:")
            print(f"  Total timesteps: {n_timesteps}")
            print(f"  Keypoints per frame: {n_keypoints}")
            print(f"  Showing timesteps: {min(timesteps_to_show, n_timesteps)}")
            
            # Create subplots for multiple timesteps
            fig = plt.figure(figsize=(15, 3 * min(timesteps_to_show, n_timesteps)))
            
            for t in range(min(timesteps_to_show, n_timesteps)):
                frame = sequence[t].reshape(n_keypoints, 3)
                
                ax = fig.add_subplot(min(timesteps_to_show, n_timesteps), 
                                    1, t + 1, projection='3d')
                
                # Plot keypoints
                ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], 
                          c=np.arange(n_keypoints), cmap='viridis', 
                          s=50, alpha=0.8)
                
                # Add text labels for first few keypoints
                for i in range(min(5, n_keypoints)):
                    ax.text(frame[i, 0], frame[i, 1], frame[i, 2], 
                           f'{i}', size=8)
                
                # Connect some keypoints (assuming basic skeleton structure)
                # This is a simple connectivity pattern - adjust based on your data
                if n_keypoints >= 17:  # Common pose formats have 17+ keypoints
                    # Example connections for a simple skeleton
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # spine
                        (1, 5), (5, 6), (6, 7),  # left arm
                        (1, 8), (8, 9), (9, 10),  # right arm
                        (0, 11), (11, 12), (12, 13),  # left leg
                        (0, 14), (14, 15), (15, 16),  # right leg
                    ]
                    
                    for conn in connections:
                        if conn[0] < n_keypoints and conn[1] < n_keypoints:
                            ax.plot([frame[conn[0], 0], frame[conn[1], 0]],
                                   [frame[conn[0], 1], frame[conn[1], 1]],
                                   [frame[conn[0], 2], frame[conn[1], 2]],
                                   'b-', alpha=0.3, linewidth=1)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'Timestep {t}')
                
                # Set equal aspect ratio
                max_range = np.array([
                    frame[:, 0].max() - frame[:, 0].min(),
                    frame[:, 1].max() - frame[:, 1].min(),
                    frame[:, 2].max() - frame[:, 2].min()
                ]).max() / 2.0
                
                mid_x = (frame[:, 0].max() + frame[:, 0].min()) * 0.5
                mid_y = (frame[:, 1].max() + frame[:, 1].min()) * 0.5
                mid_z = (frame[:, 2].max() + frame[:, 2].min()) * 0.5
                
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            plt.show()

def main():
    """Main function to run the enhanced dataset exploration"""
    
    # Define the path to Dataset0
    dataset_path = Path("/home/madjel/Projects/gitpackages/motion-foundation-model/datasets/dataset0")
    
    print("ENHANCED 3D POSE DATASET EXPLORER")
    print("="*60)
    print(f"Dataset path: {dataset_path}")
    
    # Check if directory exists
    if not dataset_path.exists():
        print(f"\nError: Directory '{dataset_path}' not found!")
        return
    
    # Define file paths
    info_data_path = dataset_path / "infoData.npy"
    h5_path = dataset_path / "time_sequences.h5"
    
    # Check what files exist
    print(f"\nChecking for data files:")
    print(f"  infoData.npy: {'✓' if info_data_path.exists() else '✗'}")
    print(f"  time_sequences.h5: {'✓' if h5_path.exists() else '✗'}")
    
    # Load and analyze metadata
    if info_data_path.exists():
        print("\n" + "="*60)
        print("LOADING METADATA")
        print("="*60)
        info_data = np.load(info_data_path, allow_pickle=True)
        if isinstance(info_data, np.ndarray) and info_data.dtype == object:
            info_data = info_data.item()
        
        # Analyze metadata structure
        analyze_metadata_structure(info_data)
    
    # Load and analyze pose data
    pose_data = None
    if h5_path.exists():
        pose_data, dataset_name = explore_h5_detailed(h5_path)
        
        if pose_data is not None:
            # Visualize sample sequences
            visualize_pose_sequence(pose_data, sample_idx=0, timesteps_to_show=3)
    else:
        print("\n" + "="*60)
        print("NOTE: time_sequences.h5 NOT FOUND")
        print("="*60)
        print("\nThe actual pose data is stored in time_sequences.h5")
        print("Without this file, we can only analyze the metadata.")
        print("\nTo get the full dataset with pose sequences:")
        print("1. Check if the file exists in another location")
        print("2. Download the complete dataset from the source")
        print("3. The h5 file should contain the actual 3D pose coordinates")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    if pose_data is not None:
        print("\nSummary for Motion Foundation Model Development:")
        print("-" * 40)
        if len(pose_data.shape) == 3:
            n_samples, n_timesteps, n_features = pose_data.shape
            if n_features % 3 == 0:
                n_keypoints = n_features // 3
                print(f"Dataset Structure:")
                print(f"  - {n_samples} motion sequences")
                print(f"  - {n_timesteps} timesteps per sequence")
                print(f"  - {n_keypoints} 3D keypoints per frame")
                print(f"\nReady for self-supervised training with:")
                print(f"  - Noise injection corruption")
                print(f"  - Random joint masking (up to {n_keypoints} joints)")
                print(f"  - Temporal masking (up to {n_timesteps} frames)")
                print(f"  - Rotation augmentations")
                print(f"  - Spatial offsets")

if __name__ == "__main__":
    main()
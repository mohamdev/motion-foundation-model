# Motion Foundation Model

LSTM-based 3D skeleton reconstruction from corrupted motion capture sequences using self-supervised learning.

## Project Overview

This project implements a motion foundation model that uses LSTM networks to reconstruct clean, height-normalized 3D skeleton sequences from corrupted ones (with noise and missing joints). The model is trained in a self-supervised manner using pose data corruption and reconstruction.

**Core Architecture:**
- Bidirectional LSTM with 3 layers (default 256 hidden units per direction)
- Input/output: flattened 3D joint coordinates (T×K×3 → T×F where F=K*3)
- Height-normalized skeleton data centered on `midHip` reference joint
- Supports variable joint counts (K=16 for F=50 datasets, K=20 for F=62 datasets)
- Gradient accumulation wrapper for effective large batch training
- Optional mixed precision (AMP) for GPU memory efficiency

**Dataset Structure:**
- Multiple `dataset*/` folders containing pose sequences
- Each dataset has two files:
  - `infoData.npy`: metadata (subjects, scale factors)
  - `time_sequences.h5`: 3D pose data stored as (N, T, F) where N=sequences, T=60 timesteps, F=features
- Feature dimension F can be 50 (48 joint coords + 2 height/weight) or 62 (60 joint coords + 2 height/weight)
- Data layout auto-detection supports both interleaved (x,y,z,x,y,z,...) and stacked (xxx...yyy...zzz...) formats
- Training scripts filter to F==62 datasets only by default

## Skeleton Joint Configuration

20 joints (used when F=62):
```
Neck, RShoulder, LShoulder, RHip, LHip, midHip,
RKnee, LKnee, RAnkle, LAnkle, RHeel, LHeel,
RSmallToe, LSmallToe, RBigToe, LBigToe,
RElbow, LElbow, RWrist, LWrist
```

The skeleton edges define connectivity for visualization and bone-length validation during layout detection.

## Common Commands

### Training

Train LSTM joint corrector on all F==62 datasets:
```bash
python lstm_reconstruct_skeleton.py \
  --datasets_root ./datasets \
  --datasets -1 \
  --epochs 40 \
  --batch-size 32 \
  --lr 1e-3 \
  --hidden 256 \
  --layers 3 \
  --dropout 0.3 \
  --noise-std-cm 2.0 \
  --save-prefix lstm_joint_corrector
```

Train with memory optimization (mixed precision + gradient accumulation):
```bash
python lstm_reconstruct_skeleton.py \
  --datasets_root ./datasets \
  --datasets -1 \
  --mixed-precision \
  --accum-steps 4 \
  --merge-mode sum \
  --batch-size 16
```

Key training flags:
- `--datasets N`: Use first N dataset folders (-1 = all)
- `--mixed-precision`: Enable AMP (float16 compute, fp32 output)
- `--accum-steps N`: Gradient accumulation over N steps
- `--merge-mode {sum,concat}`: BiLSTM merge (sum saves memory vs concat)
- `--joint-drop-prob`: Probability of masking individual joints (corruption)
- `--frame-drop-frac`: Fraction of frames to mask (temporal corruption)
- `--l2`: L2 regularization coefficient

### Visualization

Visualize clean dataset sequences with MeshCat:
```bash
python visualize_features_meshcat.py \
  --dataset-root ./datasets/dataset0 \
  --subject 3 \
  --seq 0 \
  --fps 30
```

Play all sequences for a subject:
```bash
python visualize_features_meshcat.py \
  --dataset-root ./datasets/dataset0 \
  --subject 3 \
  --all \
  --loop
```

Visualize LSTM reconstruction (white=reference, red=corrupted, green=reconstructed):
```bash
python visualize_reconstruction.py \
  --dataset-root ./datasets/dataset0 \
  --subject 3 \
  --seq -1 \
  --weights lstm_joint_corrector_best.weights.h5 \
  --model-json lstm_joint_corrector.json \
  --kmax 20
```

Explore dataset metadata and structure:
```bash
python dataset_metadata_viewer.py
```

## Architecture Details

### Data Preprocessing Pipeline

1. **Strip height/weight**: Remove last 2 columns if F % 3 == 2
2. **Layout detection**: Auto-detect interleaved vs stacked via bone-length variance scoring
3. **Reference centering**: Subtract `midHip` coordinates from all joints
4. **Height normalization**: Divide by (max_y - min_y) to normalize skeleton scale

### Corruption Strategy (Self-Supervised Training)

The model learns to denoise and fill missing data via:
- **Gaussian noise**: Added to normalized coordinates (default: mean=0cm, std=2cm, scaled by 1.7m height)
- **Joint masking**: Random joints set to zero with probability `joint_drop_prob`
- **Frame masking**: Contiguous spans of frames zeroed (default: 30% of frames in 15-frame chunks)
- **Loss masking**: Per-timestep loss weighted by fraction of visible joints

### Model Architecture Components

- **AccumModel**: Gradient accumulation wrapper that averages gradients over N steps before applying
- **Mixed precision**: Uses `mixed_float16` policy with float32 output head for numerical stability
- **BiLSTM merge modes**:
  - `concat` (default): Preserves full capacity but uses more memory (hidden*2)
  - `sum`: Saves memory but reduces model capacity
- **Regularization**: L2 on kernel and recurrent weights, dropout within LSTM layers

### Inference with Variable Joint Counts

When visualizing or running inference:
- Model is trained with fixed `Kmax` (e.g., 20 joints)
- Dataset sequences may have fewer joints (e.g., K=16)
- Sequences are padded to Kmax for model input, then cropped back to K for rendering
- Only real joints (not padding) are displayed in visualization

## Key Files

- `lstm_reconstruct_skeleton.py`: Main training script with multi-dataset support
- `visualize_reconstruction.py`: MeshCat visualization of model predictions vs ground truth
- `visualize_features_meshcat.py`: MeshCat visualization of raw dataset sequences
- `visualize_features_noisy_meshcat.py`: Visualization with noise/corruption applied
- `dataset_metadata_viewer.py`: Explore dataset structure and metadata
- `lstm_joint_corrector.json`: Saved model architecture (Keras JSON format)
- `lstm_joint_corrector.weights.h5` / `lstm_joint_corrector_best.weights.h5`: Model weights

## Data Format Notes

**HDF5 structure** (`time_sequences.h5`):
- Dataset path: `data/features`
- Shape: (N_sequences, T_timesteps=60, F_features)
- F can be 50 or 62 (only 62 used for training by default)

**Metadata structure** (`infoData.npy`):
- Python dict with keys: `subjects`, `scalefactors`
- Subject-based train/val split to avoid data leakage

## Development Patterns

When modifying the model:
1. Changes to joint configuration (NAMES/EDGES) must be consistent across all scripts
2. SEQ_LEN is hardcoded to 60 frames in multiple places
3. Layout detection uses bone-length variance; verify with new skeleton topologies
4. Mixed precision requires float32 output activation for stable loss computation
5. Gradient accumulation requires special handling in custom train_step (see AccumModel)

When adding new corruption techniques:
- Implement in Corruptor class (lstm_reconstruct_skeleton.py) or DataAugmenter class (visualize_reconstruction.py)
- Update make_sequence_pairs to apply new corruption
- Consider adding command-line flags for hyperparameter tuning
- Ensure corruption is reproducible with seed parameter

When working with datasets:
- Always probe dataset shape before loading to filter by F
- Handle both interleaved and stacked layouts via auto-detection
- Verify reference marker (midHip) exists before centering
- Check for NaN/Inf values after preprocessing transformations

# RAP

Implementation of the paper:

**RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing**

This repository provides the evaluation/pruning component of RAP.
It removes the rendering engine, Gaussian model, and scene manager from the original 3DGS implementation and enables direct pruning from a PLY file without rendering.

The code can be used as a lightweight plug-and-play module for:

- 3DGS primitive pruning
- Compression preprocessing
- Efficient transmission pipelines

---

## Requirements

- Python 3.x (recommended 3.11)
- CUDA Toolkit (recommended 12.x)
- PyTorch (must match CUDA version)

---

## Installation

Create environment:

```bash
conda create -n rap python=3.11
conda activate rap
```

Install PyTorch (example for CUDA 12.x):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Optional dependencies (recommended)

If you want GPU KNN acceleration (cuVS):

```bash
pip install cupy-cuda12x
pip install cuvs-cu12 --extra-index-url=https://pypi.nvidia.com
```

If cuVS cannot be installed, use CPU-only dependencies:

```bash
pip install plyfile scipy
```

---

## Usage

### Arguments

- `--ply_path` (required)  
  Path to input Gaussian PLY file.

- `--output_ply_path` (required)  
  Path to output pruned PLY file.

- `--keep_percent` (required)  
  Percentage of primitives to retain (0¨C1).  
  Example: `0.8` keeps 80% Gaussians.

- `--input_dim` (optional, default=15)  
  Input feature dimension.

- `--knn_k` (optional, default=128)  
  Number of neighbors for local statistics.

- `--knn_method` (optional)  
  KNN backend:
  - `ivf` ¡ª fastest GPU approximate search
  - `brute_force` ¡ª exact GPU search
  - `ckdtree` ¡ª CPU exact search (recommended if CUDA/cuVS unavailable)

- `--net_weights_path` (optional)  
  Path to pretrained network weights.  
  Default: `net_weights/net_f15.pth`.

- `--sh_degree` (optional, default=3)  
  Spherical harmonics degree.

- `--data_device` (optional, default=`cuda:0`)  
  CUDA device index.

---

## Example

```bash
python prune_percent.py \
    --ply_path /path/to/input.ply \
    --output_ply_path /path/to/output.ply \
    --keep_percent 0.8
```

---

## Notes

- This repo focuses on importance prediction and pruning only.
- Rendering-based evaluation is intentionally removed.
- Designed for fast post-hoc 3DGS compression and analysis.

---

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{yang2025rap,
  title={RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing},
  author={Yang, Kaifa and Yang, Qi and Xu, Yiling and Li, Zhu},
  booktitle={CVPR},
  year={2026}
}
```

---

# Learning SiamMAE

A from-scratch PyTorch re-implementation of [**SiamMAE**](https://arxiv.org/abs/2305.14344) (Siamese Masked Autoencoders for Video, Gupta et al., NeurIPS 2023), built as a learning project to understand the architecture in depth rather than as a research contribution.

This project started with a broader goal (active visual exploration), but that direction was dropped early on. What's left, and what this repo now documents, is a working SiamMAE: an encoder-decoder model that takes a **past frame** (fully visible) and a **future frame** (heavily masked), and learns to reconstruct the masked future patches by cross-attending to the past frame's representation. Trained on synthetic camera trajectories around rotating 3D solids, it correctly reconstructs future frames from motion context.

## What's implemented

- Vision Transformer encoder/decoder built from scratch (patchify, multi-head attention, sinusoidal 2D positional embeddings) — no `timm` or HuggingFace model code.
- Siamese encoding: shared-weight encoder run once on the unmasked past frame and once on the masked future frame.
- Asymmetric masking with a random masking ratio, plus a masking-ratio curriculum that anneals from a start to a target ratio over training.
- Cross-attention decoder that reconstructs masked future patches by attending to past-frame tokens.
- `[CLS]` token on both branches (kept masking-free) for downstream linear probing.
- Synthetic data pipeline: procedurally rendered trajectories of rotating Platonic solids (PyVista), packed into WebDataset shards for training.

## Installation

Clone the repository and install the project in editable mode:

```bash
pip install -e .
```

## Data Generation

Data generation is a two-step pipeline: render raw trajectories, then pack them into WebDataset shards.

**1. Render raw trajectories** of 3D Platonic solids under controlled camera motion:

```bash
python src/generate_poly_dataset.py \
  --num_trajs 100 \
  --length 20 \
  --resolution 224 \
  --output_dir ./data/raw/train \
  --shape icosahedron
```

Available shapes: `tetrahedron`, `cube`, `octahedron`, `dodecahedron`, `icosahedron`, `mixed` (randomly samples from all shapes).

Each trajectory is saved as per-frame JPEGs plus `.actions.npy` (camera velocities), `.states.npy` (camera states), and `.meta.json` (shape identity).

Other options:
- `--monochromatic`: render white meshes with edges only, instead of face-colored meshes.
- `--repeated_vel`: use a repeated-velocity trajectory pattern instead of a closed random-walk loop.

**2. Pack into WebDataset shards:**

```bash
python src/build_wds.py \
  --raw_dir ./data/raw/train \
  --output_prefix ./data/wds/train/platonic \
  --max_count 125
```

## Training

Point at a config file:

```bash
python src/train.py --config training_configs/laptop_test.yaml
```

Model architecture, masking schedule, optimizer, and data paths are all set via YAML, parsed into a `SiamMAEConfig` dataclass (`src/config.py`). To resume a run:

```bash
python src/train.py --config training_configs/laptop_test.yaml --ckpt_path path/to/checkpoint.ckpt --resume_id <wandb_run_id>
```

## Monitoring

Metrics and sample reconstructions (past / future / masked / reconstructed grids) are logged to Weights & Biases. To authenticate a new environment:

```bash
wandb login
```

## Code layout

- `src/config.py` — `SiamMAEConfig`, a single dataclass holding model, data, and training hyperparameters.
- `src/model.py` — the model itself: patchifier, multi-head attention, encoder/decoder blocks, `SiamMAEEncoder`, `SiamMAEDecoder`.
- `src/system.py` — `SiamMAESystem`, the PyTorch Lightning module wiring the model to the training/validation loop and optimizer.
- `src/datamodule.py` — WebDataset-backed `PlatonicDataModule`.
- `src/callbacks.py` — WandB callback for logging reconstruction image grids.
- `src/generate_poly_dataset.py`, `src/build_wds.py` — synthetic data generation and packing.

## Stack

- **Core**: PyTorch / PyTorch Lightning
- **Tracking**: Weights & Biases (WandB)
- **Config**: YAML / Python Dataclasses
- **Data Generation**: PyVista, NumPy

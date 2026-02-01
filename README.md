# Active SiamMAE

Unsupervised Active Visual Exploration using SiamMAE and Generalized Velocity.

## Installation

Clone the repository and install the project in editable mode:

```bash
pip install -e .
```

## Project Structure

- `src/`: Core logic (Model, DataModule, Lightning System)
- `training_configs/`: YAML files for experiment hyperparameters
- `data/`: Directory for .npz or .npy dataset shards
- `checkpoints/`: Model weights and training snapshots

## Data Generation

Generate synthetic datasets of 3D Platonic solids with controlled camera trajectories:

```bash
python generate_poly_dataset.py \
  --num_trajs 100 \
  --shard_size 50 \
  --length 20 \
  --resolution 224 \
  --output_dir ./data/medium_data \
  --shape icosahedron
```

### Available Shapes

- `tetrahedron`
- `cube`
- `octahedron`
- `dodecahedron`
- `icosahedron`
- `mixed` (randomly samples from all shapes)

### Generation Options

- `--num_trajs`: Number of trajectories to generate
- `--shard_size`: Trajectories per shard file
- `--length`: Steps per trajectory
- `--resolution`: Image resolution (square)
- `--monochromatic`: Use white meshes with edges only
- `--repeated_vel`: Use repeated velocity trajectory pattern

Each trajectory consists of rendered images, camera actions (velocity), camera states, and shape identifiers, saved as compressed `.npz` files.

## Training

To start training, point to a configuration file:

```bash
python src/train.py --config training_configs/laptop_test.yaml
```

## Monitoring

Metrics are synced to Weights & Biases. To authenticate a new environment:

```bash
wandb login
```

## Stack

- **Core**: PyTorch / PyTorch Lightning
- **Tracking**: Weights & Biases (WandB)
- **Config**: YAML / Python Dataclasses
- **Data Generation**: PyVista, NumPy
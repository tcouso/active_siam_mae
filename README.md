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


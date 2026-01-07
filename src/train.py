import argparse
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodule import PlatonicDataModule
from src.system import LitPlatonicSystem
from src.dataset import read_platonic_solids_dataset

# TODO: Understand input dimensions for the model
# TODO: Understand forward and backward pass logic

def main(args):
    full_dataset = read_platonic_solids_dataset(args.data_dir)
    dm = PlatonicDataModule(full_dataset, batch_size=args.batch_size)

    # Placeholder architecture - ensure input dim matches your data resolution (e.g., 64x64)
    backbone = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256*256*3, 512),
        nn.ReLU(),
        nn.Linear(512, 5) 
    )
    
    system = LitPlatonicSystem(backbone, lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.ckpt_dir,
        filename='platonic-{epoch:02d}-{val_loss:.2f}'
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback]
    )

    trainer.fit(system, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/medium_data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    
    args = parser.parse_args()
    main(args)
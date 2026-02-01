import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import ActSiamMAEConfig
from src.datamodule import PlatonicDataModule
from src.system import ActSiamMAESystem
from src.dataset import read_platonic_solids_dataset

def main(args):
    config = ActSiamMAEConfig.from_yaml(args.config)    
    pl.seed_everything(config.seed, workers=True)

    run_name = f"ActSiamMAE_mask{config.masking_ratio}_dim{config.hidden_dim}_lay{config.encoder_num_layers}"
    wandb_logger = WandbLogger(
            project="ActiveSiamMAE", 
            name=run_name,
            log_model="all" 
        )
    wandb_logger.experiment.config.update(vars(config))

    full_dataset = read_platonic_solids_dataset(config.data_dir)
    data_module = PlatonicDataModule(
        full_dataset, 
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        num_workers=config.num_workers,
        seed=config.seed
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"./checkpoints/{wandb_logger.version}",
        filename="sample-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy if config.devices != 1 else "auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
    )
    system = ActSiamMAESystem(config)
    trainer.fit(system, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    main(parser.parse_args())

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from src.model import ActSiamMAEConfig, ActiveSiamMAEEncoder, ActiveSiamMAEDecoder


class ActiveSiamMAESystem(pl.LightningModule):
    def __init__(self, config: ActSiamMAEConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config        
        self.encoder = ActiveSiamMAEEncoder(config)
        self.decoder = ActiveSiamMAEDecoder(config)
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        return optim

    def training_step(self, batch, batch_idx):
        # TODO: 
        # 1. Get past/future frames
        # 2. Run forward pass
        # 3. Calculate Loss (MSE on invisible patches only)
        pass
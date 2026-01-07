import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class LitPlatonicSystem(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # Auto-logs params
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        # Unpack dictionary batch
        images = batch['images'].float() # Ensure float for NN
        actions = batch['actions']
        
        # Example logic: Predict something from images
        preds = self(images) 
        
        # Dummy loss (Replace with your actual loss)
        loss = F.mse_loss(preds, actions) 
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



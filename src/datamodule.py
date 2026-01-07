import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split


class PlatonicDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        total = len(self.dataset)
        train_len = int(0.8 * total)
        val_len = total - train_len
        self.train_ds, self.val_ds = random_split(self.dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

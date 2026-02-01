import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from src.dataset import PlatonicSample


def platonic_solids_transform(sample: PlatonicSample) -> PlatonicSample:
    sample["images"] = sample["images"].float() / 255.0
    sample["actions"] = sample["actions"].float()
    sample["states"] = sample["states"].float()
    return sample


class PlatonicDataModule(pl.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int, train_ratio: float, num_workers: int=4, seed: int=42):
        super().__init__()
        self.dataset = dataset
        self.dataset.transform = platonic_solids_transform
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str = None) -> None:
        total = len(self.dataset)
        train_len = int(self.train_ratio * total)
        val_len = total - train_len
        self.train_ds, self.val_ds = random_split(
            self.dataset, 
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

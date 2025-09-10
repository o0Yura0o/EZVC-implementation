import torch, torch.nn as nn, lightning as L
from torch.utils.data import DataLoader
from models import LSTMInfiller
from build_dataset import MelTokenDataset

class LitMelInfiller(L.LightningModule):
    def __init__(self, **kw):
        super().__init__()
        self.net = LSTMInfiller(**kw)
        self.l1 = nn.L1Loss()

    def training_step(self, batch, _):
        mel, tokens = batch["mel"], batch["tokens"]   # [B,T,80], [B,:]
        T = mel.size(1) // 2
        mel_pre, mel_tar = mel[:, :T], mel[:, T:]
        tok_suf        = tokens[:, T]              
        mel_hat = self.net(mel_pre, tok_suf)
        loss = self.l1(mel_hat, mel_tar)
        self.log("train/l1", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4)
    
class MelTokenDataModule(L.LightningDataModule):
    def __init__(
        self,
        wav_list,
        xeus,
        kmeans,
        mel_cfg: dict,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.wav_list = wav_list
        self.xeus = xeus
        self.kmeans = kmeans
        self.mel_cfg = mel_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.predict_set = None

    def setup(self):
        self.train_set = MelTokenDataset(self.wav_list, self.xeus, self.kmeans, self.mel_cfg)
        self.val_set = self.train_set
        self.test_set = self.train_set
        self.predict_set = self.train_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
    
#Training loop not prepared yet
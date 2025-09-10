import torch, torch.nn as nn, lightning as L
from models import LSTMInfiller

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
    
# DataModule
#wav -> mel、xeus feats -> kmeans tokens
# TO DO:
#1.build a k-means model using features in extract_units.py
#2.Generate mel of a set of sound data
#3.Get their embeddings and discrete units
#4.The prepared dataset has the components: mel, discrete tokens

import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa

class MelTokenDataset(Dataset):
    def __init__(self, wav_list, xeus, kmeans, mel_cfg):
        self.wav_list = wav_list
        self.xeus = xeus
        self.kmeans = kmeans
        self.mel_cfg = mel_cfg

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_path = self.wav_list[idx]
        wav, sr = sf.read(wav_path)

        # 1) Mel
        mel = librosa.feature.melspectrogram(
            y=wav, sr=sr,
            n_fft=self.mel_cfg["n_fft"],    #400
            hop_length=self.mel_cfg["hop_length"],  #160
            n_mels=self.mel_cfg["n_mels"]   #80
        )
        mel = torch.from_numpy(librosa.power_to_db(mel)).T  # [T, 80]

        # 2) XEUS embedding -> kmeans tokens
        with torch.no_grad():
            feats = self.xeus.encode(
                torch.tensor(wav[None], dtype=torch.float32).cuda(),
                torch.tensor([len(wav)]).cuda()
            )[0][13]  # 第14層 [B,T,H]
        tokens = torch.tensor(self.kmeans.predict(feats.cpu().numpy()))  # [T]

        return {
            "mel": mel,         # [T, 80]
            "tokens": tokens    # [T]
        }
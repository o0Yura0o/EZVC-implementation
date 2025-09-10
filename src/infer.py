from models import LSTMInfiller
import torch

def load_mel():
    pass

def load_tokens():
    pass

net = LSTMInfiller.load_state_dict(torch.load("ckpt/mel_lstm.pt")) ; net.eval()
mel = load_mel("data/wavs/test.wav")                   # [T,80]
tokens = load_tokens("data/wavs/test.tokens.npy")      # [T]
T = mel.shape[0] // 2
mel_hat_suf = net(mel[:T][None], torch.tensor(tokens[:])[None]).squeeze(0)  # [T2,80]

# vocoder (HiFi-GAN)
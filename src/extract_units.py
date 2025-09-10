import torch, soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from espnet2.tasks.ssl import SSLTask

from sklearn.cluster import MiniBatchKMeans
import joblib, numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
xeus, xeus_args = SSLTask.build_model_from_file(
    # "XEUS/model/config.yaml",
    None,
    "XEUS/model/xeus_checkpoint_old.pth",
    device,
)

# print(xeus)

#read dataset
# for 迴圈
wav, sr = sf.read("data/wavs/test1.wav")       # XEUS 期望 16kHz
wavs = pad_sequence(torch.tensor([wav]), batch_first=True).to(device)
wav_lengths = torch.LongTensor([len(wav)]).to(device)

# 取最後一層 hidden states => [B, T, H]
feats = xeus.encode(wavs, wav_lengths, use_final_output=False)[0][13]

print(feats)
# 假設先把多檔 feats 蒐集起來到 feat_pool: [N_frames, H]
feat_pool = ""

kmeans = MiniBatchKMeans(n_clusters=500, batch_size=2048, n_init='auto').fit(feat_pool)
joblib.dump(kmeans, "src/kmeans.joblib")

# 推理量化
def quantize(feats, kmeans):
    # feats: [T, H] -> token ids: [T]
    c = kmeans.predict(feats.detach().cpu().numpy())
    return torch.from_numpy(c).long()
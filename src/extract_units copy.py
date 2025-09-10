# src/extract_units.py（節錄）
import torch, soundfile as sf
from espnet2.bin.s2t_inference import Speech2Text

model = Speech2Text.from_pretrained(
  "espnet/xeus"
)

print(model)

speech, rate = sf.read("data/wavs/test1.wav")
text, *_ = model(speech)[0]
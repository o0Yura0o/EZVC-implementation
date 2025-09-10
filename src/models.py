import torch, torch.nn as nn

class LSTMInfiller(nn.Module):
    def __init__(self, mel_dim=80, token_vocab=500, token_dim=128, cond_dim=256, hidden=512, layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(token_vocab, token_dim)
        self.mel_in = nn.Linear(mel_dim, cond_dim)
        self.fuse = nn.Linear(cond_dim + token_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.proj = nn.Linear(hidden, mel_dim)

    def forward(self, mel_prefix, tokens_suffix):
        """
        mel_prefix:  [B, T1, 80]   # 前半段 Mel
        tokens_suffix: [B, T2]     # 整段tokens
        產生: 預測的 mel_suffix_hat: [B, T2, 80]
        """
        # 把 prefix 壓成條件向量（也可改 cross-attention）
        cond = self.mel_in(mel_prefix)                  # [B,T1,cond_dim]
        cond = cond.mean(dim=1, keepdim=True)           # [B,1,cond_dim] 簡化做全局條件

        tok = self.token_emb(tokens_suffix)             # [B,T2,token_dim]
        cond_expand = cond.expand(-1, tok.size(1), -1)  # [B,T2,cond_dim]
        x = torch.cat([cond_expand, tok], dim=-1)       # [B,T2,cond+tok]
        x = self.fuse(x)
        y, _ = self.lstm(x)
        mel_hat = self.proj(y)
        return mel_hat

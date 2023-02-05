import torch
import torch.nn as nn

class MultiHeadSelfAtten(nn.Module):

    def __init__(self, dim=512, head=8) -> None:
        super().__init__()
        self.dim = dim
        self.head = head
        self.qkv_proj = nn.Linear(dim, dim*3)
        self.out_proj = nn.Linear(dim, dim)

    # [B, S, D] -> [B, H, S, D/H]
    def reshape(self, x):
        return x.reshape(x.shape[0], -1, self.head, self.dim // self.head).transpose(1, 2)

    def self_attn(self, q, k, v, mask):
        # q, k, v: [B, H, S, D/H]
        # out: [B, H, S, D/H]
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5) 
        score = score.masked_fill(mask == 0, -1e3)
        attn = score.softmax(dim=-1)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x, mask):
        # x: [B, S, D]
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        batch = q.shape[0]
        q, k, v = self.reshape(q), self.reshape(k), self.reshape(v)
        out = self.self_attn(q, k, v, mask)
        # merge heads: [B, H, S, D/H] -> [B, S, D]
        out = out.transpose(1, 2).reshape(batch, -1, self.dim)

        out = self.out_proj(out)
        return out

def test_torch(dtype):
    m = MultiHeadSelfAtten()
    m.eval()
    with torch.autocast('cuda', dtype):
        m.to('cuda:1')
        _ = m(torch.rand(1, 32, 512).to('cuda:1'), torch.rand(1, 32, 32).to('cuda:1'))
# test_torch(torch.float32)
#test_torch(torch.float16)

import tops
tops.benchmark(MultiHeadSelfAtten(), (torch.rand(1, 32, 512), torch.rand(1, 32, 32)), verbose=True, logdir="./logs/multi-atten-1")
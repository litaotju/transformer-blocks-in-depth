import torch
import torch.nn as nn

class SelfAtten(nn.Module):

    def __init__(self, dim=512) -> None:
        super().__init__()
        self.dim = dim
        self.qkv_proj = nn.Linear(dim, dim*3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        # x: [B, S, D]
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        # [B, S, S]
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5) 
        score = score.masked_fill(mask == 0, -1e3)
        attn = score.softmax(dim=-1)
        # [B, S, D]
        out = torch.matmul(attn, v)

        out = self.out_proj(out)
        return out

def test_torch(dtype):
    m = SelfAtten()
    m.eval()
    with torch.autocast('cuda', dtype):
        m.to('cuda:1')
        _ = m(torch.rand(1, 32, 512).to('cuda:1'), torch.rand(1, 32, 32).to('cuda:1'))

if __name__ == "__main__":
    test_torch(torch.float32)
    import tops
    tops.benchmark(SelfAtten(), (torch.rand(1, 32, 512), torch.rand(1, 32, 32)), verbose=True, logdir="./logs/self-atten-1")
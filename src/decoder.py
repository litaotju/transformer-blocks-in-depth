import torch
import torch.nn as nn
from multihead import MultiHeadSelfAtten

class DecoderLayer(nn.Module):

    def __init__(self, dim = 512, head = 8) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAtten(dim, head)
        self.norm = nn.LayerNorm(dim)
        self.mlp_linear1 = nn.Linear(dim, dim*4)
        self.act= nn.GELU()
        self.mlp_linear2 = nn.Linear(dim*4, dim)

    def forward(self, x, mask):
        # x: [B, S, D]
        # mask: [B, S, S]
        # z: [B, S, D]
        y = self.attn(x, mask)
        x = x + self.norm(y)
        z = self.mlp_linear2(self.act(self.mlp_linear1(x)))
        z = y + self.norm(z)
        return z

def test_torch():
    model = DecoderLayer()
    model.eval()
    model.to('cuda:1')
    _ = model(torch.rand(1, 32, 512).to('cuda:1'), torch.rand(1, 32, 32).to('cuda:1'))

if __name__ == "__main__":
    # test_torch()
    import tops
    tops.benchmark(DecoderLayer(), (torch.rand(1, 32, 512), torch.rand(1, 32, 32)), verbose=True, logdir="./logs/decoder-1")
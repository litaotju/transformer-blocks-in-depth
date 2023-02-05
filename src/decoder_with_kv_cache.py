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
        # q: [B, H, S, D/H] or [B, H, 1, D/H]
        # k, v: [B, H, S, D/H]
        # out: [B, H, S, D/H] or [B, H, 1, D/H]
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5) 
        score = score.masked_fill(mask == 0, -1e3)
        attn = score.softmax(dim=-1)
        out = torch.matmul(attn, v)
        return out

    def forward(self, x, mask, past_keys=None, past_values=None):
        # Non-cached state
        #   x: [B, S, D], past_keys, past_values: None
        #
        # Cached:
        #   past_keys and past_values are not None, they are of shape [B, H, S, D/H]
        #   x: [B, 1, D], means "query" is only one token.
        #   thus the hidden state passing to next step is only [B, 1, D]
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = self.reshape(q), self.reshape(k), self.reshape(v)

        batch = q.shape[0]
        if past_keys is not None:
            # concat in time dimension
            k = torch.cat((past_keys, k), dim=-2)
            v = torch.cat((past_values, v), dim=-2)

        out = self.self_attn(q, k, v, mask)
        # merge heads: [B, H, S, D/H] -> [B, S, D]
        out = out.transpose(1, 2).reshape(batch, -1, self.dim)

        out = self.out_proj(out)
        return out, k, v


class DecoderLayer(nn.Module):

    def __init__(self, dim = 512, head = 8) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAtten(dim, head)
        self.norm = nn.LayerNorm(dim)
        self.mlp_linear1 = nn.Linear(dim, dim*4)
        self.act= nn.GELU()
        self.mlp_linear2 = nn.Linear(dim*4, dim)

    def forward(self, x, mask, past_keys=None, past_values=None):
        # x: [B, S, D]
        # mask: [B, S, S]
        # z: [B, S, D]
        y, k, v = self.attn(x, mask, past_keys, past_values)
        x = x + self.norm(y)
        z = self.mlp_linear2(self.act(self.mlp_linear1(x)))
        z = y + self.norm(z)
        return z, k, v


if __name__ == "__main__":
    state, mask = torch.rand(1, 1, 512).to('cuda:1'), torch.rand(1, 33, 33).to('cuda:1')
    past_keys, past_values = torch.rand(1, 8, 32, 64).to('cuda:1'), torch.rand(1, 8, 32, 64).to('cuda:1')

    model = DecoderLayer()
    model.eval()
    model.to('cuda:1')
    # _ = model(state, mask, past_keys, past_values)

    import tops
    tops.benchmark(model, (state, mask, past_keys, past_values), verbose=True, logdir="./logs/decoder-kv-cache")
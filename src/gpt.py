import torch
from torch import nn
from typing import List
from dataclasses import dataclass
from decoder_with_kv_cache import DecoderLayer

@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    max_seq_len: int

class GPT(nn.Module):

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(config.d_model, config.n_heads) for _ in range(config.n_layers)])
    
    def forward(self, x, mask, past_keys: List[torch.Tensor], past_values: List[torch.Tensor]):
        keys = []
        values = [] 
        for i, decoder in enumerate(self.decoders):
            if past_keys is not None:
                x, current_key, current_value = decoder(x, mask, past_keys[i], past_values[i])
            else:
                x, current_key, current_value = decoder(x, mask, None, None)
            keys.append(current_key)
            values.append(current_value)
        return x, keys, values

if __name__ == "__main__":
    device = 'cuda:1'
    small = GPTConfig(vocab_size=5000, n_layers=12, n_heads=8, d_model=768, d_ff=3072, dropout=0.1, max_seq_len=1024)
    # memic generation stage
    gpt_small = GPT(small).to(device)

    bs = 8
    context_len = 32
    prompt =  torch.rand(bs, context_len, small.d_model).to(device)
    x = prompt
    keys = None
    values = None

    iteration = 10
    for i in range(iteration):
        # TODO: why the mask is [1, 1, 1, seq] here
        # can the generation stage take no mask?
        # since it's auto regressive, the current token should attend to all previous tokens
        # seq = 32 + i + 1
        # mask = torch.ones(1, 1, 1, seq).to('cuda:1')
        print("iteration: ", i)
        if keys is not None:
            print("key shape", keys[0].shape)
        mask = None
        x, keys, values = gpt_small(x, mask, keys, values)
        x = x[:, -1:, :]
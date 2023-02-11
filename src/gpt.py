import torch
from torch import nn
import torch.nn.functional as F

import math
import sys
from typing import List
from dataclasses import dataclass

class MultiHeadSelfAtten(nn.Module):

    def __init__(self, dim, head) -> None:
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
        score = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.dim)) 

        # Equal to 
        # score = torch.where(mask, score, -1e3)
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


class DecoderBlock(nn.Module):

    def __init__(self, dim, head, dim_ff) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAtten(dim, head)
        self.norm1 = nn.LayerNorm(dim)

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_linear1 = nn.Linear(dim, dim_ff)
        self.act= nn.GELU()
        self.mlp_linear2 = nn.Linear(dim_ff, dim)

    def forward(self, hidden, mask, past_keys=None, past_values=None):
        '''
         hidden: [B, S, D]
         mask: [B, S, S]
        '''
        # gpt-2 and gpt-3 use layer norm before block, https://github.com/karpathy/minGPT
        x = self.norm1(hidden)
        x, k, v = self.attn(x, mask, past_keys, past_values)
        x = hidden + x
        x = x + self.mlp_linear2(self.act(self.mlp_linear1(self.norm2(x))))
        return x, k, v


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
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.decoders = nn.ModuleList([DecoderBlock(config.d_model, config.n_heads, config.d_ff) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.register_buffer("mask", 
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len))
    
    def forward(self, idx:torch.Tensor, past_keys: torch.Tensor, past_values: torch.Tensor):
        '''
        input shapes:
            idx: [B, S]
            past_keys: [B, H, S, D/H]
            past_values: [B, H, S, D/H]
        output shapes:
            logits: [B, 1]
            past_keys: [B, H, S+1, D/H]
            past_values: [B, H, S+1, D/H]
        '''
        keys = []
        values = []

        # In the first step, past_keys and past_values are None, idx is [B, length of prompt]
        # And in the following steps, past_keys and past_values are not None, idx is [B, 1]

        ## TODO: how to avoid this if-else such that only one onnx is needed for both
        ## context stage, and generation stage?
        if past_keys is None:
            pos = torch.arange(idx.shape[1], dtype=torch.long).to(idx.device)
        else:
            # if past_keys is not None, the current token's position is the length of past keys
            context_len = past_keys[0].shape[-2]
            pos = torch.tensor(context_len, dtype=torch.long).to(idx.device)

        x = self.word_embedding(idx)
        x += self.pos_embedding(pos).unsqueeze(0)

        T_end = idx.size(1) + (past_keys[0].size(-2) if past_keys is not None else 0)
        T_start = T_end - idx.size(1)
        mask = self.mask[:, :, T_start:T_end, :T_end]

        for i, decoder in enumerate(self.decoders):
            if past_keys is not None:
                x, current_key, current_value = decoder(x, mask, past_keys[i], past_values[i])
            else:
                x, current_key, current_value = decoder(x, mask, None, None)
            keys.append(current_key)
            values.append(current_value)

        x = self.ln(x)
        # only last token is used for prediction
        logits = self.lm_head(x[:, -1:, :])

        keys = torch.stack(keys) 
        values = torch.stack(values)
        return logits, keys, values

    def generate(self, prompt):
        x = prompt
        keys = None
        values = None
        iteration = 10

        generated = []
        for i in range(iteration):
            logits, keys, values = self(x, keys, values)

            # only last token is used for prediction
            logits = logits[:, -1:, :]
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            x = torch.multinomial(probs[:,-1,:], num_samples=1)
            generated.append(x)
        return torch.cat(generated, dim=1)

configs = {
    #Just for quick test
    'tiny' : GPTConfig(vocab_size=50257, n_layers=1, n_heads=2, d_model=768, d_ff=3072, dropout=0.1, max_seq_len=2048),
    # model configs from the original GPT-3 paper
    'small' : GPTConfig(vocab_size=50257, n_layers=12, n_heads=12, d_model=768, d_ff=3072, dropout=0.1, max_seq_len=2048),
    'medium' : GPTConfig(vocab_size=50257, n_layers=24, n_heads=16, d_model=1024, d_ff=4096, dropout=0.1, max_seq_len=2048),
    'large' : GPTConfig(vocab_size=50257, n_layers=24, n_heads=16, d_model=1536, d_ff=1536*4, dropout=0.1, max_seq_len=2048),
    'xl' : GPTConfig(vocab_size=50257, n_layers=24, n_heads=24, d_model=2048, d_ff=2048*4, dropout=0.1, max_seq_len=2048),
    '2.7B' : GPTConfig(vocab_size=50257, n_layers=32, n_heads=32, d_model=2560, d_ff=2560*4, dropout=0.1, max_seq_len=2048),
    '6.7B': GPTConfig(vocab_size=50257, n_layers=32, n_heads=32, d_model=4096, d_ff=4096*4, dropout=0.1, max_seq_len=2048),
    '13B': GPTConfig(vocab_size=50257, n_layers=40, n_heads=40, d_model=5140, d_ff=5140*4, dropout=0.1, max_seq_len=2048),
    '175B' :GPTConfig(vocab_size=50257, n_layers=96, n_heads=96, d_model=12288, d_ff=49152, dropout=0.1, max_seq_len=2048)
}

def export_model(config: GPTConfig, model_path: str, device: str):
    bs = 8
    context_len = 32
    past_shape = (config.n_layers, bs, config.n_heads, context_len, config.d_model//config.n_heads)
    past_keys = torch.randn(past_shape).to(device)
    past_values = torch.randn(past_shape).to(device)
    idx = torch.randint(config.vocab_size, [bs, 1]).to(device)

    gpt = GPT(config).to(device)
    torch.onnx.export(
        gpt,
        (idx, past_keys, past_values),
        model_path,
        input_names=['idx', 'past_keys', 'past_values'],
        output_names=['logits', 'past_keys', 'past_values'],
        dynamic_axes={
            'idx': {0: 'batch_size', 1: 'seq_len'},
            'past_keys': {1: 'batch_size', 3: 'seq_len'},
            'past_values': {1: 'batch_size', 3: 'seq_len'},
        },
        verbose=True,
    )

def test_tops(config: GPTConfig):
    bs = 8
    context_len = 32
    past_shape = (config.n_layers, bs, config.n_heads, context_len, config.d_model//config.n_heads)
    past_keys = torch.randn(past_shape).to(device)
    past_values = torch.randn(past_shape).to(device)
    idx = torch.randint(config.vocab_size, [bs, 1]).to(device)

    gpt = GPT(config).to(device)
 
    import tops
    tops.benchmark(gpt, (idx, past_keys, past_values), 1, verbose=True, logdir="./logs/gpt",
        input_names=['idx', 'past_keys', 'past_values'],
        output_names=['logits', 'past_keys', 'past_values'],
        dynamic_axes={
            'idx': {0: 'batch_size', 1: 'seq_len'},
            'past_keys': {1: 'batch_size', 3: 'seq_len'},
            'past_values': {1: 'batch_size', 3: 'seq_len'},
        },
    )

def test_inference(config: GPTConfig):
    gpt_small = GPT(config).to(device)
    bs = 8
    context_len = 32
    prompt =  torch.randint(config.vocab_size, [bs, context_len]).to(device)
    generated = gpt_small.generate(prompt)
    print("generated shape:", generated.shape)

if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'tiny'
    device = 'cuda:1'
    config = configs[config_name]
    print(config)

    # test_inference(config)
    export_model(config, f'gpt_{config_name}.onnx', device)
import torch
from torch import nn
import tops

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)

tops.benchmark(transformer_model, (src, tgt), verbose=True, logdir="./logs/transformer-1")
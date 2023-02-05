import torch
import torch.nn as nn

norm = nn.LayerNorm(512).to('cuda')

# x = norm(torch.rand(1, 32, 512).to('cuda'))

import tops
tops.benchmark(norm, (torch.rand(1, 32, 512).to('cuda'),), verbose=True, logdir="./logs/layernorm-1")
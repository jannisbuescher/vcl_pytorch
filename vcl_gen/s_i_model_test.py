import torch

from s_i_model import SIVAE

model = SIVAE(10, 10, 10, 2, 10)

input = torch.rand((64, 10))
output = model(input, 0)

sample = model.sample(4, 0)

import torch
from model import VariationalMLP

model = VariationalMLP(10, 10, 10, 3)

input = torch.rand((64, 10))
output = model(input)
print(output)

params = model.copy_params()
print(len(params))
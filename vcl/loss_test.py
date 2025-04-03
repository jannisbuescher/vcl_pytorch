import torch
from model import VariationalMLP

from loss import loss, kl_div


model = VariationalMLP(10, 10, 10, 3)
model2 = VariationalMLP(10, 10, 10, 3)

params = model.copy_params()
params2 = model2.copy_params()

print(kl_div(params, params))
print(kl_div(params, params2))

input = torch.rand((3, 10))
output = model(input)

targets = torch.tensor([1, 3, 7])

l = loss(output, targets, params, params)
print(l)
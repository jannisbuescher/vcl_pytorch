import torch
from model import VAE

model = VAE(10, 10, 10, 2, 5)

input = torch.rand((64, 10))
output, mu, logsigma = model(input, 0)
output2, _, _ = model(input, 2)

model.get_params_grad_on()
model.copy_params()
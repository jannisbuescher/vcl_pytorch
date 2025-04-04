import torch

from loss import loss as loss_fn
from model import VAE

model = VAE(10, 10, 10, 2, 5)
model2 = VAE(10, 10, 10, 2, 5)

input = torch.rand((64, 10))
output, mu, logsigma = model(input, 0)
output2, mu2, logsigma2 = model(input, 2)

params = model.get_params_grad_on()
params2 = model2.get_params_grad_on()
model.copy_params()

loss = loss_fn(output, input, mu, logsigma, params, params2)
print(loss)

loss = loss_fn(output, output, mu, logsigma, params, params)
print(loss)

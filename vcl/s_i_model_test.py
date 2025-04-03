import torch
from s_i_model import SIMLP

model = SIMLP(10, 10, 10, 3)
model2 = SIMLP(10, 10, 10,3)

params = model.parameters()
params2 = model2.parameters()

input = torch.rand((64,10))
output = model(input)

print(output)

loss = torch.nn.MSELoss()(output, torch.zeros_like(output))
loss.backward()

grads = model.get_gradients()
print(grads)
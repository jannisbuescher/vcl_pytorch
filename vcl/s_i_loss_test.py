import torch

from s_i_loss import si_loss, compute_omega

from s_i_model import SIMLP

model = SIMLP(10,10, 10, 3)
model2 = SIMLP(10,10, 10, 3)
models = [SIMLP(10, 10, 10, 3) for _ in range(10)]
models2 = [SIMLP(10, 10, 10, 3) for _ in range(10)]

param = [a for a in model.parameters()]
param2 = [a for a in model2.parameters()]
params = [[a for a in model.parameters()] for model in models]
params2 = [[a for a in model.parameters()] for model in models2]

omegas = params2[:-1]

Omega = compute_omega(params, omegas)

input = torch.rand((64,10))
out = model(input)

loss = si_loss(out, torch.randint(0, 10, (64,)), param, param2, Omega)

print(loss)
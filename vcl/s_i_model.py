import torch
import torch.nn as nn
import torch.functional as F


class SIMLP(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim)]

        for _ in range(nb_layers - 2):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return torch.softmax(x, dim=-1)
    
    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grads.append(p.grad)
        return grads
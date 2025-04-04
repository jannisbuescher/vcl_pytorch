import torch
import torch.nn as nn
import torch.functional as F

from typing import List


class SIEncoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hid_dim)]

        for _ in range(nb_layers - 2):
            layers.append(nn.Linear(hid_dim, hid_dim))

        self.layers = nn.ModuleList(layers)
        self.mu = nn.Linear(hid_dim, out_dim)
        self.logsigma = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        return mu, torch.clamp(logsigma, min=-10, max=10)
    
    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grads.append(p.grad)
        return grads
    
class SIDecoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        super().__init__()

        self.heads = nn.ModuleList([nn.Linear(in_dim, hid_dim) for _ in range(nb_heads)])

        layers = []
        for _ in range(nb_layers - 2):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, idx: int):
        x = self.heads[idx](x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return torch.sigmoid(x)
    
    def get_gradients(self) -> List[torch.Tensor]:
        grads = []
        for layer in self.layers:
            for p in layer.parameters():
                grads.append(p.grad)
        return grads
    
    def get_params(self) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            for p in layer.parameters():
                params.append(p)
        return params
    

class SIVAE(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        super().__init__()
        self.encoders = nn.ModuleList([SIEncoder(in_dim, hid_dim, out_dim, nb_layers) for _ in range(nb_heads)])
        self.decoder = SIDecoder(out_dim, hid_dim, in_dim, nb_layers, nb_heads)
        self.z_dim = out_dim

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        mu, logsigma = self.encoders[idx](x)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = mu + eps * torch.exp(0.5 * logsigma)
        x = self.decoder(z, idx)
        return x, mu, logsigma
    
    def sample(self, nb_samples: int, idx:int) -> torch.Tensor:
        device = self.decoder.layers[0].weight.device
        z = torch.normal(torch.zeros((nb_samples, self.z_dim)), torch.ones((nb_samples, self.z_dim))).to(device)
        x = self.decoder(z, idx)
        return x
    
    def get_gradients(self) -> List[torch.Tensor]:
        return self.decoder.get_gradients()
    
    def get_params(self) -> List[torch.Tensor]:
        return self.decoder.get_params()
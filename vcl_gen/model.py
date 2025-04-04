import torch
import torch.nn as nn
import torch.functional as F

from vcl.model import VariationalLinear

from typing import List


class VariationalEncoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        layers = [VariationalLinear(in_dim, hid_dim)]

        for _ in range(nb_layers - 2):
            layers.append(VariationalLinear(hid_dim, hid_dim))
        
        heads_mu = []
        heads_logsigma = []
        for _ in range(nb_heads):
            heads_mu.append(VariationalLinear(hid_dim, out_dim))
            heads_logsigma.append(VariationalLinear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.heads_mu = nn.ModuleList(heads_mu)
        self.heads_logsigma = nn.ModuleList(heads_logsigma)

    def forward(self, x: torch.Tensor, idx: int=0) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        mu = self.heads_mu[idx](x)
        logsigma = self.heads_logsigma[idx](x)
        return mu, logsigma
    
    def copy_params(self, idx: int=0) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.copy_params())
        params.append(self.heads_mu[idx].copy_params())
        params.append(self.heads_logsigma[idx].copy_params())
        return params

    def get_params_grad_on(self, idx: int=0) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.get_params_grad_on())
        params.append(self.heads_mu[idx].get_params_grad_on())
        params.append(self.heads_logsigma[idx].get_params_grad_on())
        return params


class VariationalDecoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        heads = []
        for _ in range(nb_heads):
            heads.append(VariationalLinear(in_dim, hid_dim))

        layers = []
        for _ in range(nb_layers - 2):
            layers.append(VariationalLinear(hid_dim, hid_dim))
        
        layers.append(VariationalLinear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor, idx: int=0) -> torch.Tensor:
        x = self.heads[idx](x)
        x = torch.relu(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1]
        return torch.sigmoid(x)
    
    def copy_params(self, idx: int=0) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.copy_params())
        params.append(self.heads_mu[idx].copy_params())
        params.append(self.heads_logsigma[idx].copy_params())
        return params

    def get_params_grad_on(self, idx: int=0) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params.append(layer.get_params_grad_on())
        params.append(self.heads_mu[idx].get_params_grad_on())
        params.append(self.heads_logsigma[idx].get_params_grad_on())
        return params
        

class VAE(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        self.encoders = nn.ModuleList([VariationalEncoder(in_dim, hid_dim, out_dim, nb_layers, 1) for _ in range(nb_heads)])
        self.decoder = VariationalDecoder(out_dim, hid_dim, in_dim, nb_layers, nb_heads)

    def forward(self, x: torch.Tensor, idx: int):
        mu, logsigma = self.encoder[idx](x)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = mu + eps * torch.exp(0.5 * logsigma)
        x = self.decoder(z, idx)
        return x
    
    def copy_params(self, idx):
        return self.decoder.copy_params(idx)
    
    def get_params_grad_on(self, idx):
        return self.decoder.get_params_grad_on(idx)
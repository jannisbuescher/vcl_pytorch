import torch
import torch.nn as nn
import torch.functional as F

from typing import List, Tuple


class VariationalLinear(nn.Module):
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_mu = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.W_logsigma = nn.Parameter(torch.zeros(in_dim, out_dim))

        self.b_mu = nn.Parameter(torch.zeros(out_dim,))
        self.b_logsigma = nn.Parameter(torch.zeros(out_dim,))

        with torch.no_grad():
            self.W_mu.normal_(mean=0, std=0.1)
            self.W_logsigma.normal_(mean=-12.0, std=0.1)

            self.b_mu.normal_(mean=0, std=0.1)
            self.b_logsigma.normal_(mean=-12.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps_weights = torch.normal(torch.zeros_like(self.W_mu), torch.ones_like(self.W_mu))
        eps_bias = torch.normal(torch.zeros_like(self.b_mu), torch.ones_like(self.b_mu))

        weights = self.W_mu + eps_weights * torch.exp(0.5 * self.W_logsigma)
        bias = self.b_mu + eps_bias * torch.exp(0.5 * self.b_logsigma)

        return x @ weights + bias
    
    def get_params_grad_on(self) -> Tuple[torch.Tensor]:
        return self.W_mu, self.W_logsigma, self.b_mu, self.b_logsigma,
    
    def copy_params(self) -> Tuple[torch.Tensor]:
        return self.W_mu.clone().detach(), self.W_logsigma.clone().detach(), self.b_mu.clone().detach(), self.b_logsigma.clone().detach()


class VariationalEncoder(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        super().__init__()
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
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        mu = self.heads_mu[idx](x)
        logsigma = self.heads_logsigma[idx](x)
        return mu, torch.clamp(logsigma, min=-10, max=10)
    
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
        super().__init__()
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
        x = self.layers[-1](x)
        return torch.sigmoid(x)
    
    def copy_params(self, idx: int=0) -> List[List[torch.Tensor]]:
        params = []
        for layer in self.layers:
            params.append(layer.copy_params())
        if idx != -1:
            params.append(self.heads[idx].copy_params())
        return params

    def get_params_grad_on(self, idx: int=0) -> List[List[torch.Tensor]]:
        params = []
        for layer in self.layers:
            params.append(layer.get_params_grad_on())
        if idx != -1:
            params.append(self.heads[idx].get_params_grad_on())
        return params
        

class VAE(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, nb_heads: int):
        super().__init__()
        self.encoders = nn.ModuleList([VariationalEncoder(in_dim, hid_dim, out_dim, nb_layers, 1) for _ in range(nb_heads)])
        self.decoder = VariationalDecoder(out_dim, hid_dim, in_dim, nb_layers, nb_heads)
        self.z_dim = out_dim

    def forward(self, x: torch.Tensor, idx: int):
        mu, logsigma = self.encoders[idx](x)
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(mu))
        z = mu + eps * torch.exp(0.5 * logsigma)
        x = self.decoder(z, idx)
        return x, mu, logsigma
    
    def sample(self, nb_samples: int, idx:int) -> torch.Tensor:
        device = self.decoder.layers[0].W_mu.device
        z = torch.normal(torch.zeros((nb_samples, self.z_dim)), torch.ones((nb_samples, self.z_dim))).to(device)
        x = self.decoder(z, idx)
        return x
    
    def copy_params(self, idx: int = 0) -> List[List[torch.Tensor]]:
        return self.decoder.copy_params(idx)
    
    def get_params_grad_on(self, idx: int = 0) -> List[List[torch.Tensor]]:
        return self.decoder.get_params_grad_on(idx)
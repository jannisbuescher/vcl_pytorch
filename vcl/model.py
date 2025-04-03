import torch

import torch.nn as nn
import torch.functional as F

from typing import List

from copy import deepcopy


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
            self.b_logsigma.normal_(mean=-13.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps_weights = torch.normal(torch.zeros_like(self.W_mu), torch.ones_like(self.W_mu))
        eps_bias = torch.normal(torch.zeros_like(self.b_mu), torch.ones_like(self.b_mu))

        weights = self.W_mu + eps_weights * torch.exp(self.W_logsigma)
        bias = self.b_mu + eps_bias * torch.exp(self.b_logsigma)

        return x @ weights + bias
    
    def get_params_grad_on(self):
        return self.W_mu, self.W_logsigma, self.b_mu, self.b_logsigma,
    
    def copy_params(self):
        return self.W_mu.clone().detach(), self.W_logsigma.clone().detach(), self.b_mu.clone().detach(), self.b_logsigma.clone().detach()
    

class VariationalMLP(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, nb_layers: int, heads: int = 0):
        super().__init__()
        layers = [VariationalLinear(in_dim, hid_dim)]

        for _ in range(nb_layers - 2):
            layers.append(VariationalLinear(hid_dim, hid_dim))

        if heads == 0:
            self.has_heads = False
            layers.append(VariationalLinear(hid_dim, out_dim))
        else:
            self.has_heads = True
            heads = []
            for _ in range(heads):
                heads.append(VariationalLinear(hid_dim, out_dim))
            self.heads = nn.ModuleList(heads)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, head_id: int = -1) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        if head_id != -1:
            x = self.heads[head_id](x)
        return torch.softmax(x, dim=-1)

    def copy_params(self, head_id: int = -1) -> List:
        params = []
        for layer in self.layers:
            params.append(layer.copy_params())
        if head_id != -1:
            params.append(self.heads[head_id].copy_params())
        return params
    
    def copy_head_params(self, head_id: int):
        return self.heads[head_id].copy_params()
    
    def get_params_grad_on(self, head_id: int = -1):
        params = []
        for layer in self.layers:
            params.append(layer.get_params_grad_on())
        if head_id != -1:
            params.append(self.heads[head_id].get_params_grad_on())
        return params
    
    def copy(self):
        in_dim, hid_dim = self.layers[0].W_mu.shape
        out_dim = self.layers[-1].W_mu.shape[1]
        nb_layers = len(self.layers)
        if self.has_heads:
            heads = len(self.heads)
        else:
            heads = 0
        model = VariationalMLP(in_dim, hid_dim, out_dim, nb_layers, heads)
        model.load_state_dict(deepcopy(self.state_dict()))
        return model
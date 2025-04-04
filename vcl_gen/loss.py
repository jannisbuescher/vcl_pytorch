import torch
import torch.nn as nn
import torch.functional as F

from typing import List


def vae_loss(x_recon: torch.Tensor,
            x: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    # bce = -torch.sum(x * torch.log(x_recon + 1e-8) + (1 - x) * torch.log(1 - x_recon + 1e-8))
    bce = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')

    kl_div = -0.5 * torch.sum(1 + logsigma - torch.square(mu) - torch.exp(logsigma))

    return bce + kl_div


def loss(x_recon: torch.Tensor,
         x: torch.Tensor,
         mu: torch.Tensor,
         logsigma: torch.Tensor,
         params: List[List[torch.Tensor]],
         params_prior: List[List[torch.Tensor]],
         kl_weight : float = 1 / (6000)
) -> torch.Tensor:
    return vae_loss(x_recon, x, mu, logsigma) + kl_weight* kl_div(params, params_prior)
    


def kl_div(params_posterior: List[List[torch.Tensor]], params_prior: List[List[torch.Tensor]]):
    def kl(mu_q, mu_p, logsigma_q, logsigma_p):
        term1 = torch.sum(logsigma_p - logsigma_q)
        term2 = torch.sum(torch.square(mu_p - mu_q) / torch.exp(logsigma_p))
        term3 = torch.sum(torch.exp(logsigma_q) / torch.exp(logsigma_p))
        term4 = -mu_q.shape[0]
        return term1 + term2 + term3 + term4
    
    kl_div_loss = 0
    for post, prior in zip(params_posterior, params_prior):
        kl_div_loss += kl(post[0].flatten(), prior[0].flatten(), post[1].flatten(), prior[1].flatten())

    return kl_div_loss
import torch
import torch.nn as nn
import torch.functional as F

from typing import List

def loss(logits: torch.Tensor,
         targets: torch.Tensor,
         params_posterior: List[List[torch.Tensor]],
         params_prior: List[List[torch.Tensor]],
         kl_weight: float = 1/60000,
):
    nll = nll_loss(logits, targets)
    kl_div_loss = kl_div(params_posterior, params_prior)
    return nll + kl_weight * kl_div_loss



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

def nll_loss(logits: torch.Tensor, targets:torch.Tensor):
    log_probs = torch.log(logits + 1e-9)
    return torch.sum(-torch.gather(log_probs, 1, targets.unsqueeze(1)).squeeze(1))
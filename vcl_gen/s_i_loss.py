import torch
import torch.nn as nn
import torch.functional as F

from typing import List, Dict, Iterator

from loss import vae_loss

from copy import deepcopy

def si_loss(x_recon: torch.Tensor,
            x: torch.Tensor,
            mu: torch.Tensor,
            logsigma: torch.Tensor,
            params: List[nn.Parameter],
            prev_parameters: List[List[nn.Parameter]],
            omegas: List[List[nn.Parameter]],
            c: float = 0.1
) -> torch.Tensor:
    vae_term = vae_loss(x_recon, x, mu, logsigma)
    prev_params = prev_parameters[-1]
    Omega = compute_omega(prev_parameters, omegas)
    si_term = si(params, prev_params, Omega)
    return vae_term + c * si_term

def si(
    params: List[nn.Parameter],
    prev_params: List[nn.Parameter],
    omega: List[nn.Parameter]
) -> torch.Tensor:
    si = 0
    if len(omega) > 0:
        for i in range(len(params)):
            p = params[i]
            prev_p = prev_params[i]
            Omega = omega[i]
            si += torch.sum(Omega * torch.square(p - prev_p))
    return si


def compute_omega(
        prev_parameters: List[List[nn.Parameter]],
        omegas: List[List[nn.Parameter]]
) -> List[torch.Tensor]:
    Omega = []
    for task_id in range(len(prev_parameters) - 1):
        param_nu = prev_parameters[task_id + 1]
        param_nu_prev = prev_parameters[task_id]
        omega_nu = omegas[task_id]
        for i, (param_nu_i, param_nu_prev_i, omega_nu_i) in enumerate(zip(param_nu, param_nu_prev, omega_nu)):
            delta = torch.square(param_nu_i - param_nu_prev_i)
            if len(Omega) <= i:
                Omega.append(omega_nu_i / (delta + 1e-5))
            else:
                Omega[i] += omega_nu_i / (delta + 1e-5)
    return Omega

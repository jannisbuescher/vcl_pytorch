import torch
import torch.nn as nn

from typing import List


def ewc_loss(outputs: torch.Tensor,
             targets: torch.Tensor,
             params: List[torch.Tensor],
             optimal_params: List[List[torch.Tensor]],
             fisher_matrices: List[List[torch.Tensor]],
             lambda_reg: float = 100,
):
    loss = nn.functional.cross_entropy(outputs, targets)

    ewc_loss = 0
    for optimal_params_i, fisher_matrices_i in zip(optimal_params, fisher_matrices):
        for param, optimal_param, fisher_matrix in zip(params, optimal_params_i, fisher_matrices_i):
            ewc_loss += (fisher_matrix * (param - optimal_param)**2).sum()

    return loss + (lambda_reg / 2) * ewc_loss
import torch
import torch.nn as nn
import torch.functional as F

from model import VariationalMLP
from loss import loss as loss_fn
from permutedmnist import create_data_loaders
from coresets import get_coreset_dataloader

from typing import List, Tuple

from torch.utils.data import DataLoader

def get_prior_params_like(params: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    prior_params = []
    for layer in params:
        param_W_mu, param_W_logsigma, param_b_mu, param_b_logsigma = layer
        prior_params.append((torch.zeros_like(param_W_mu, requires_grad=False), 
                                torch.zeros_like(param_W_logsigma, requires_grad=False),
                                torch.zeros_like(param_b_mu, requires_grad=False), 
                                torch.zeros_like(param_b_logsigma, requires_grad=False),
                                ))
    return prior_params


def validation(model: VariationalMLP, dataloaders: List[Tuple[DataLoader]], nb_tasks: int, verbose: bool=True):
    model.eval()

    for task_id_eval, (_, testloader) in enumerate(dataloaders[:nb_tasks]):
        correct = 0
        total = 0
        for im, target in testloader:
            out = model(im)
            correct += (out.argmax(dim=1) == target).sum()
            total += target.shape[0]
        if verbose:
            print(f"On Task {task_id_eval} got acc: {(correct / total):.4f}")

def training(model: VariationalMLP,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             prior: List[List[torch.Tensor]],
             num_epochs:int,
             verbose: bool=True
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for im, targets in dataloader:
            opti.zero_grad()

            out = model(im)
            loss = loss_fn(out, targets, model.get_params_grad_on(), prior)
            
            loss.backward()
            opti.step()

            total_loss += loss.item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    return model


def train(model: VariationalMLP, num_epochs: int, num_tasks: int, coresets: bool = False, num_epochs_coreset: int = 100):
    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = create_data_loaders(256, num_tasks)

    if coresets:
        coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=200)

    prior_params = get_prior_params_like(model.copy_params())


    for task_id, (trainloader, _) in enumerate(dataloaders):

        model = training(model, opti, trainloader, prior_params, num_epochs)

        validation(model, dataloaders, nb_tasks=task_id+1)

        # swap priors
        prior_params = model.copy_params()


        if coresets:
            print("Coreset fine-tune")
            model_finetune = model.copy()
            opti_finetune = torch.optim.Adam(model_finetune.parameters(), 0.001)

            model_finetune = training(model_finetune, 
                                      opti_finetune, 
                                      coreset_dataloaders[task_id],
                                      prior_params,
                                      num_epochs_coreset,
                                      verbose=False
            )

            validation(model_finetune, dataloaders, task_id+1)



if __name__ == '__main__':

    model = VariationalMLP(28*28, 100, 10, 3)
    train(
        model,
        num_epochs=3,
        num_tasks=5,
        coresets=True,
        num_epochs_coreset=10,
    )

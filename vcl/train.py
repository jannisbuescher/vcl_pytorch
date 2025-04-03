import torch
import torch.nn as nn
import torch.functional as F

from tqdm import tqdm

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


def validation(
        model: VariationalMLP,
        dataloaders: List[Tuple[DataLoader]],
        nb_tasks: int,
        verbose: bool=True,
        device: str='cpu'
) -> float:
    model.eval()
    avg_acc = 0
    for task_id_eval, (_, testloader) in enumerate(dataloaders[:nb_tasks]):
        correct = 0
        total = 0
        for im, target in testloader:
            im = im.to(device)
            target = target.to(device)
            out = model(im)
            correct += (out.argmax(dim=1) == target).sum()
            total += target.shape[0]
        avg_acc += correct / total
        if verbose:
            print(f"On Task {task_id_eval} got acc: {(correct / total):.4f}")
    return avg_acc / nb_tasks

def training(model: VariationalMLP,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             prior: List[List[torch.Tensor]],
             num_epochs:int,
             verbose: bool=True,
             use_tqdm: bool=False,
             device: str='cpu'
):
    model.train()

    if use_tqdm:
        logger = tqdm
        verbose = False
    else:
        logger = lambda x: x

    for epoch in logger(range(num_epochs)):
        total_loss = 0

        for im, targets in dataloader:
            im = im.to(device)
            targets = targets.to(device)

            opti.zero_grad()

            out = model(im)
            loss = loss_fn(out, targets, model.get_params_grad_on(), prior)
            
            loss.backward()
            opti.step()

            total_loss += loss.cpu().item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    return model


def train(
        model: VariationalMLP,
        num_epochs: int,
        num_tasks: int,
        coresets: bool = False,
        coreset_size: int = 200,
        num_epochs_coreset: int = 100,
        verbose: bool = False
):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = create_data_loaders(256, num_tasks)

    if coresets:
        coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=coreset_size)

    prior_params = get_prior_params_like(model.copy_params())

    avg_accs = []
    avg_accs_finetune = []


    for task_id, (trainloader, _) in enumerate(dataloaders):

        model = training(model, opti, trainloader, prior_params, num_epochs, use_tqdm=not verbose, verbose=verbose, device=device)

        avg_acc = validation(model, dataloaders, nb_tasks=task_id+1, device=device)
        avg_accs.append(avg_acc)

        # swap priors
        prior_params = model.copy_params()


        if coresets:
            print("Coreset fine-tune")
            model_finetune = model.copy().to(device)
            opti_finetune = torch.optim.Adam(model_finetune.parameters(), 0.001)

            model_finetune = training(model_finetune, 
                                      opti_finetune, 
                                      coreset_dataloaders[task_id],
                                      prior_params,
                                      num_epochs_coreset,
                                      verbose=False,
                                      device=device
            )

            avg_acc_finetune = validation(model_finetune, dataloaders, task_id+1, device=device)
            avg_accs_finetune.append(avg_acc_finetune)
    if coresets:
        return avg_accs, avg_accs_finetune
    else:
        return avg_accs



if __name__ == '__main__':

    model = VariationalMLP(28*28, 100, 10, 3)
    res = train(
        model,
        num_epochs=3,
        num_tasks=3,
        coresets=True,
        num_epochs_coreset=10,
        verbose=False
    )
    print(res)

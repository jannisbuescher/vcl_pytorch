import torch
import torch.nn as nn
import torch.functional as F

from s_i_model import SIMLP
from s_i_loss import si_loss as loss_fn
from loss import nll_loss
from permutedmnist import create_data_loaders
from coresets import get_coreset_dataloader

from typing import List, Tuple

from torch.utils.data import DataLoader

def get_prior_params_from(model: SIMLP) -> List[List[torch.Tensor]]:
    prior_params = model.parameters()
    return [[param.clone().detach() for param in prior_params]]


def validation(model: SIMLP, dataloaders: List[Tuple[DataLoader]], nb_tasks: int, verbose: bool=True):
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

def training(model: SIMLP,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             prior: List[List[torch.Tensor]],
             omegas: List[List[torch.Tensor]],
             num_epochs:int,
             verbose: bool=True
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        omega_this_task = None
        for im, targets in dataloader:
            opti.zero_grad()

            out = model(im)
            params = [param for param in model.parameters()]

            # calculate omega 
            nll = nll_loss(out, targets)
            nll.backward(retain_graph=True)

            grad = model.get_gradients()
            if omega_this_task is None:
                omega_this_task = grad
            else:
                omega_this_task += grad

            loss = loss_fn(out, targets, params, prior, omegas)
            
            loss.backward()
            opti.step()

            total_loss += loss.item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    omegas.append([torch.abs(omega) for omega in omega_this_task])
    prior_parameters = prior + get_prior_params_from(model)
    return model, omegas, prior_parameters


def train(model: SIMLP, num_epochs: int, num_tasks: int, coresets: bool = False, num_epochs_coreset: int = 100):
    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = create_data_loaders(256, num_tasks)

    if coresets:
        coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=200)

    prior_params = get_prior_params_from(model)
    omegas = []


    for task_id, (trainloader, _) in enumerate(dataloaders):

        model, omegas, prior_params = training(model, opti, trainloader, prior_params, omegas, num_epochs)

        validation(model, dataloaders, nb_tasks=task_id+1)


        # if coresets:
        #     print("Coreset fine-tune")
        #     model_finetune = model.copy()
        #     opti_finetune = torch.optim.Adam(model_finetune.parameters(), 0.001)

        #     model_finetune = training(model_finetune, 
        #                               opti_finetune, 
        #                               coreset_dataloaders[task_id],
        #                               prior_params,
        #                               num_epochs_coreset,
        #                               verbose=False
        #     )

        #     validation(model_finetune, dataloaders, task_id+1)



if __name__ == '__main__':

    model = SIMLP(28*28, 100, 10, 3)
    train(
        model,
        num_epochs=3,
        num_tasks=5,
        coresets=False,
        num_epochs_coreset=10,
    )

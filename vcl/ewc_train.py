import torch
import torch.nn as nn
import torch.functional as F

from ewc_model import EWC
from ewc_loss import ewc_loss as loss_fn
from loss import nll_loss
from permutedmnist import create_data_loaders
from coresets import get_coreset_dataloader

from typing import List, Tuple

from torch.utils.data import DataLoader

from tqdm import tqdm

def get_prior_params_from(model: EWC) -> List[List[torch.Tensor]]:
    prior_params = model.parameters()
    return [[param.clone().detach() for param in prior_params]]


def validation(model: EWC,
               dataloaders: List[Tuple[DataLoader]],
               nb_tasks: int,
               verbose: bool=True,
               device: str = 'cpu'
):
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

def training(model: EWC,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             optimal_params: List[List[torch.Tensor]],
             fisher_matrices: List[List[torch.Tensor]],
             num_epochs:int,
             verbose: bool=True,
             use_tqdm: bool = False,
             device: str= 'cpu'
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
            params = [param for param in model.parameters()]

            loss = loss_fn(out, targets, params, optimal_params, fisher_matrices)
            
            loss.backward()
            opti.step()

            total_loss += loss.cpu().item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    return model

def compute_optimal_params(model: EWC, dataloader: DataLoader):
    optimal_params = []
    for param in model.get_params():
        optimal_params.append(param.clone().detach())
    return optimal_params
    

def compute_fisher_matrices(model: EWC, dataloader: DataLoader, n_fisher_samples: int = 600):
    model.eval()

    fisher_matrices = []
    for param in model.get_params():
        fisher_matrices.append(torch.zeros_like(param, device=param.device))
    
    images = []
    for image, _ in dataloader:
        images.append(image)
        if len(torch.cat(images)) >= n_fisher_samples:
            break
    images = torch.cat(images)[:n_fisher_samples].to(device=param.device)

    for i in range(images.shape[0]):
        log_probs = nn.functional.log_softmax(model(images[i:i+1]), dim=1)
        probs = torch.exp(log_probs.detach())
        sampled_label = torch.multinomial(probs, 1).item()
        
        model.zero_grad()
        log_probs[0, sampled_label].backward()

        for i, param in enumerate(model.get_params()):
            fisher_matrices[i] += torch.square(param.grad.detach()) / n_fisher_samples

    return fisher_matrices


def train(model: EWC,
          num_epochs: int,
          num_tasks: int,
          coresets: bool = False,
          num_epochs_coreset: int = 100,
          verbose=False,
):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = create_data_loaders(256, num_tasks)

    # if coresets:
    #     coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=200)
    
    optimal_params = []
    fisher_matrices = []

    avg_accs = []

    for task_id, (trainloader, _) in enumerate(dataloaders):

        model = training(model,
                        opti,
                        trainloader,
                        optimal_params,
                        fisher_matrices,
                        num_epochs,
                        use_tqdm=not verbose,
                        verbose=verbose,
                        device=device
                        )

        avg_acc = validation(model, dataloaders, nb_tasks=task_id+1, device=device)
        avg_accs.append(avg_acc)

        # update params
        optimal_params.append(compute_optimal_params(model, trainloader))
        fisher_matrices.append(compute_fisher_matrices(model, trainloader))


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

    return avg_accs


if __name__ == '__main__':

    model = EWC(28*28, 100, 10, 3)
    avg_accs = train(
        model,
        num_epochs=3,
        num_tasks=5,
        coresets=False,
        num_epochs_coreset=10,
        verbose=True,
    )
    print(avg_accs)

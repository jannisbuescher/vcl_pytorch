import torch
import torch.nn as nn
import torch.functional as F

from s_i_model import SIVAE
from s_i_loss import si_loss as loss_fn
from loss import vae_loss
from one_digit_mnist import get_digit_dataloaders
# from coresets import get_coreset_dataloader

from typing import List, Tuple

from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

def get_prior_params_from(model: SIVAE) -> List[List[torch.Tensor]]:
    prior_params = model.get_params()
    return [[param.clone().detach() for param in prior_params]]


def sampling(
        model: SIVAE,
        nb_tasks: int,
        nb_samples: int = 4,
        save_fig: bool = False
) -> float:
    model.eval()
    f, ax = plt.subplots(nb_tasks, nb_samples)
    for task_id in range(nb_tasks):
        if nb_tasks > 1:
            ax[task_id, 0].set_ylabel(f"Task {task_id}", rotation=0, labelpad=20)
        out = model.sample(nb_samples, task_id)
        for sample in range(nb_samples):
            if nb_tasks == 1:
                ax[sample].imshow(out[sample].cpu().detach().view(28, 28), cmap='gray')
                ax[sample].set_xticks([])
                ax[sample].set_yticks([]) 
            else:
                ax[task_id, sample].imshow(out[sample].cpu().detach().view(28, 28), cmap='gray')
                ax[task_id, sample].set_xticks([])
                ax[task_id, sample].set_yticks([])
    if save_fig:
        f.tight_layout()
        f.savefig(f"graphics/si_sample{nb_tasks}.png")
    else:
        f.tight_layout()
        f.show()

def training(model: SIVAE,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             prior: List[List[torch.Tensor]],
             omegas: List[List[torch.Tensor]],
             num_epochs:int,
             task_id: int,
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
        omega_this_task = None
        for im in dataloader:
            im = im.to(device)

            opti.zero_grad()

            out, mu, logsigma = model(im, task_id)
            params = [param for param in model.get_params()]

            # calculate omega 
            vae = vae_loss(out, im, mu, logsigma)
            vae.backward(retain_graph=True)

            grad = model.get_gradients()
            if omega_this_task is None:
                omega_this_task = grad
            else:
                omega_this_task += grad

            loss = loss_fn(out, im, mu, logsigma, params, prior, omegas)
            
            loss.backward()
            opti.step()

            total_loss += loss.cpu().item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    omegas.append([torch.abs(omega) for omega in omega_this_task])
    prior_parameters = prior + get_prior_params_from(model)
    return model, omegas, prior_parameters


def train(model: SIVAE,
          num_epochs: int,
          coresets: bool = False,
          num_epochs_coreset: int = 100,
          verbose=False,
          save_fig: bool = False
):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = get_digit_dataloaders(256)

    # if coresets:
    #     coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=200)

    prior_params = get_prior_params_from(model)
    omegas = []

    avg_accs = []

    for task_id, trainloader in enumerate(dataloaders):

        model, omegas, prior_params = training(model,
                                               opti,
                                               trainloader,
                                               prior_params,
                                               omegas,
                                               num_epochs,
                                               task_id,
                                               use_tqdm=not verbose,
                                               verbose=verbose,
                                               device=device
                                                )

        sampling(model, nb_tasks=task_id+1, save_fig=save_fig)


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

    model = SIVAE(28*28, 500, 50, 2, 10)
    avg_accs = train(
        model,
        num_epochs=20,
        coresets=False,
        num_epochs_coreset=10,
        verbose=True,
        save_fig=True
    )
    print(avg_accs)

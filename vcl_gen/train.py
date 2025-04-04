import torch

from model import VAE
from loss import loss as loss_fn
from one_digit_mnist import get_digit_dataloaders

from typing import List, Tuple

from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

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


def sampling(
        model: VAE,
        dataloaders: List[Tuple[DataLoader]],
        nb_tasks: int,
        nb_samples: int = 4,
        verbose: bool=True,
        device: str='cpu',
        save_fig: bool = False
) -> float:
    model.eval()
    f, ax = plt.subplots(nb_tasks, nb_samples)
    for task_id in range(nb_tasks):
        if nb_tasks > 1:
            ax[task_id, 0].set_ylabel(f"Task {task_id}", rotation=0)
        out = model.sample(nb_samples, task_id)
        for sample in range(nb_samples):
            if nb_tasks == 1:
                ax[sample].imshow(out[sample].cpu().detach().view(28, 28))
                ax[sample].set_xticks([])
                ax[sample].set_yticks([]) 
            else:
                ax[task_id, sample].imshow(out[sample].cpu().detach().view(28, 28))
                ax[task_id, sample].set_xticks([])
                ax[task_id, sample].set_yticks([])
    if save_fig:
        f.savefig(f"graphics/sample{nb_tasks}.png")
    else:
        f.show()

def training(model: VAE,
             opti: torch.optim.Optimizer,
             dataloader: DataLoader,
             prior: List[List[torch.Tensor]],
             num_epochs:int,
             task_id: int,
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

    kl_weight = 1 / (len(dataloader) * 256)

    for epoch in logger(range(num_epochs)):
        total_loss = 0

        for im in dataloader:
            im = im.to(device)

            opti.zero_grad()

            out, mu, logsigma = model(im, task_id)
            loss = loss_fn(out, im, mu, logsigma, model.get_params_grad_on(-1), prior, kl_weight)
            
            loss.backward()
            opti.step()

            total_loss += loss.cpu().item()

        if verbose:
            print(f"Loss: {total_loss / len(dataloader):.4f}")
    return model


def train(
        model: VAE,
        num_epochs: int,
        coresets: bool = False,
        coreset_size: int = 200,
        num_epochs_coreset: int = 100,
        verbose: bool = False,
        save_fig: bool = False
):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    opti = torch.optim.Adam(model.parameters(), 0.001)

    dataloaders = get_digit_dataloaders(256)

    # if coresets:
    #     coreset_dataloaders = get_coreset_dataloader(dataloaders, 265, coreset_size=coreset_size)

    prior_params = get_prior_params_like(model.copy_params(-1))

    avg_accs = []
    avg_accs_finetune = []


    for task_id, trainloader in enumerate(dataloaders):

        model = training(model, opti, trainloader, prior_params, num_epochs, task_id, use_tqdm=not verbose, verbose=verbose, device=device)

        sampling(model, dataloaders, nb_tasks=task_id+1, device=device, save_fig=save_fig)
        # avg_accs.append(avg_acc)

        # swap priors
        prior_params = model.copy_params(-1)


        # if coresets:
        #     print("Coreset fine-tune")
        #     model_finetune = model.copy().to(device)
        #     opti_finetune = torch.optim.Adam(model_finetune.parameters(), 0.001)

        #     model_finetune = training(model_finetune, 
        #                               opti_finetune, 
        #                               coreset_dataloaders[task_id],
        #                               prior_params,
        #                               num_epochs_coreset,
        #                               verbose=False,
        #                               device=device
        #     )

        #     avg_acc_finetune = validation(model_finetune, dataloaders, task_id+1, device=device)
        #     avg_accs_finetune.append(avg_acc_finetune)
    if coresets:
        return avg_accs, avg_accs_finetune
    else:
        return avg_accs



if __name__ == '__main__':

    model = VAE(28*28, 500, 50, 2, 10)
    res = train(
        model,
        num_epochs=20,
        coresets=False,
        num_epochs_coreset=10,
        verbose=False,
        save_fig=True,
    )
    print(res)


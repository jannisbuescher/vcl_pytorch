import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from typing import List

def get_digit_dataloaders(batch_size=256) -> List[DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True)

    digit_dataloaders = []

    for digit in range(10):
        digit_subset = SingleDigitMNIST(mnist_dataset, digit)
        digit_dataloader = DataLoader(digit_subset, batch_size=batch_size, shuffle=True)
        digit_dataloaders.append(digit_dataloader)

    return digit_dataloaders


class SingleDigitMNIST(Dataset):
    def __init__(self, data: torch.Tensor, digit1: int):
        train_data = data.data
        train_mask = data.targets == digit1
        self.data = train_data[train_mask]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datapoint = (self.data[idx].float() / self.data[idx].max()).view(1, 28, 28)

        return datapoint.view(-1)
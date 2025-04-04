import torch

from one_digit_mnist import get_digit_dataloaders

import matplotlib.pyplot as plt

dataloaders = get_digit_dataloaders()

for digit, dataloader in enumerate(dataloaders):
    for im in dataloader:
        print(im.shape)
        print(im.dtype)
        plt.imsave('graphics/digit'+str(digit) + '.png', im[0].view(28, 28))
        break
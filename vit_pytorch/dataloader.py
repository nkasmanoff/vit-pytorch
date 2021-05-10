

from __future__ import print_function

import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from torch.utils.data import random_split


import sys

#from vit_pytorch.efficient import ViT


def get_CIFAR_data(dset='cifar10',
                   val_size = 5000,
                   batch_size = 64,
                   transforms=transforms.Compose([
                    transforms.RandomRotation(degrees=15),
                #  transforms.RandomVerticalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                   ])):

    if dset == 'cifar10':
        dataset = datasets.CIFAR10(root='../data/', download=True, transform=transforms)
        test_dataset = datasets.CIFAR10(root='../data/', train=False, transform=transforms)
    elif dset == 'cifar100':
        dataset = datasets.CIFAR100(root='../data/', download=True, transform=transforms)
        test_dataset = datasets.CIFAR100(root='../data/', train=False, transform=transforms)

    else:
        print("Must select cifar10 or cifar100")
        sys.exit()



    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=4, pin_memory=True)


    return train_loader, val_loader, test_loader

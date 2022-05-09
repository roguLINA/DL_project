import os
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

def set_random_seeds(seed_value=0, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_train_test_dataloaders(dataset_type='cifar10', root_data_folder='./data', batch_size=64):
    '''source https://github.com/kuangliu/pytorch-cifar/blob/master/main.py'''
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomAffine(30, shear=30), 
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ]
#     ) 
#     test_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ]
#     )
    
    data_folder = os.path.join(root_data_folder, dataset_type)
    
    if dataset_type == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, 
            download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_folder, train=False,
            download=True, transform=test_transform
        )
    elif dataset_type == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_folder, train=True, 
            download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_folder, train=False,
            download=True, transform=test_transform
        )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    
    return (train_set, test_set), (train_loader, test_loader)
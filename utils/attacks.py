# %%writefile utils/attacks.py
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)


def req_grad(model: nn.Module, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.
    :param model: pretrained network
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)


def get_image_grad_on_bacth(
    model: nn.Module,
    images: torch.tensor,
    loss: nn.Module,
    norm: str = None
):
    """Calculate gradient of loss w.r.t to images.
    :param model: pretrained network
    :param images: batch of images
    :param loss: loss function
    :param norm: use inf or l2 vector norm to get a unitary vector in a given norm.
    :return: (normed) gradients of loss w.r.t to images and (optionally)
    minimal absolute value of the gradient

    Note 1: It's supposed that images, labels and model are on the same device
    Note 2: We don't need labels to calculate loss. As our only goal is to change class
    going in some direction (ex. gradient), we always suppose that initial prediction of label
    is correct.
    """
    x_adv = torch.clone(images)
    x_adv.requires_grad = True

    logits = model(x_adv)
    loss_val = loss(logits, torch.argmax(logits, dim=1))
    loss_val.backward()

    images_grad = x_adv.grad.data
    if norm == 'inf':
        images_grad = torch.sign(images_grad)
    elif norm == 'l2':
        images_grad = images_grad / (images_grad.
                                     reshape(images_grad.shape[0], -1).
                                     norm(p=2, dim=1)[:, None, None, None])
    return images_grad


def attack_step(
    model: nn.Module,
    images: torch.tensor,
    images_grad: torch.tensor,
    image_diap: Tuple[float, float],
    eps: float
) -> torch.tensor:
    """One step of gradient attack.
    :param model: pretrained network
    :param images: batch of images
    :param images_grad: batch of gradients of loss w.r.t to images
    :param image_diap: min and max values of the images in loader, used for clipping
    :param eps: attack power
    :return: attacked images, predictions of model under attack
    """
    attacked_images = images + eps * images_grad
    attacked_images = attacked_images.clamp(*image_diap)
    logits_adv = model(attacked_images)
    _, preds_adv = torch.max(logits_adv, dim=1)

    return attacked_images, preds_adv


def fgsm_attack(
    model: nn.Module,
    loader: DataLoader,
    dataset_size: int,
    batch_size: int,
    image_diap: Tuple[float, float],
    loss: nn.Module,
    eps_list:  np.ndarray,
    device: str = 'cpu',
    norm: str = 'inf'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basic one-step gradient attack implementation.
    :param model: pretrained network
    :param loader: data loader
    :param dataset_size: number of samples in dataset
    :param batch_size: number of samples in batch
    :param image_diap: min and max values of the images in loader, used for clipping
    :param loss: loss function
    :param eps_list: list of attack power to test
    :param device: device type (cuda or cpu)
    :param norm: in what norm to make a step in the direction of the unitary vector,
    possible options 'inf'(classic fgsm, step to sign of gradient direction), 
    'l2' (step in the gradient direction)
    :return: true labels, predictions of unattacked and attacked model
    """
    model.eval()
    req_grad(model, state=False)
    n_of_exp = len(eps_list)

    targets = torch.tensor([], device=device)
    preds_attacked = torch.empty([n_of_exp, dataset_size], device=device)
    preds_unattacked = torch.tensor([], device=device)

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        images, labels = batch[0].to(device), batch[1].to(device)
        len_of_batch = labels.size()[0]
        targets = torch.cat([targets, labels.detach()])
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        preds_unattacked = torch.cat([preds_unattacked, preds.detach()])

        images_grad = get_image_grad_on_bacth(model, images, loss, norm)

        for i, eps in enumerate(eps_list):
            _, preds_adv = attack_step(model, images, images_grad, image_diap, eps)
            preds_attacked[i, idx*batch_size:idx*batch_size+len_of_batch] = preds_adv.detach()

    return (
        targets.cpu().numpy(),
        preds_attacked.cpu().numpy(),
        preds_unattacked.cpu().numpy(),
    )

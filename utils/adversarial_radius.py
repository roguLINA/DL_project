# %%writefile utils/adversarial_radius.py
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)

from .attacks import req_grad, get_image_grad_on_bacth, attack_step


def get_adversarial_radii_on_batch(
    model: nn.Module,
    len_of_batch: int,
    images: torch.tensor,
    image_diap: Tuple[float, float],
    preds_unattacked: torch.tensor,
    images_grad: torch.tensor,
    max_steps: int,
    step_size: float,
    tol: float,
    device: str
) -> Tuple[torch.tensor, torch.tensor]:
    """The algorithm to find an adversarial radii for a batch of images.
    :param model: pretrained network
    :param len_of_batch: number of samples in a batch
    :param images: batch of images
    :param image_diap: min and max values of the images in loader, used for clipping
    :param preds_unattacked: predictions of network before adversarial attack
    :param images_grad: batch of gradients of loss w.r.t to images
    :param max_steps: maximal number of algorithm steps
    :param step_size: step_size parameter of algorithm
    :param tol: maximum error of radii estimation to stop algorithm earlier
    :param device: device type (cuda or cpu)
    :return: adversarial radii, errors of estimation

    Note 1: Our confidence in convergence to the true adversarial radius is based
    on the small step_size and good properties (local convexity) of decision regions of the functions.
    But theoretically there are no guarantees that error of estimation is less than the radius iteself.
    """
    guess_left = torch.zeros(len_of_batch).to(device)
    guess_right = (torch.ones(len_of_batch) * step_size).to(device)
    adversarial_radii_ = (torch.ones(len_of_batch) * np.inf).to(device)
    tolerances_ = (torch.ones(len_of_batch) * np.inf).to(device)
    step = 0

    while True and (step <= max_steps):
        guess_right = guess_right[:, None, None, None]
        _, preds_adv = attack_step(model, images, images_grad, image_diap, guess_right)
        classified_correctly = (preds_unattacked == preds_adv.detach())
        guess_right = guess_right.squeeze()
        adversarial_radii_[~classified_correctly] = guess_right[~classified_correctly]
        tolerances_ = adversarial_radii_ - guess_left

        if torch.all(tolerances_ <= tol):
            break

        if step != max_steps:
            buffer = guess_right[classified_correctly]
            guess_right[classified_correctly] = 1.5 * guess_right[classified_correctly] - 0.5 * guess_left[classified_correctly]
            guess_left[classified_correctly] = buffer
            guess_right[~classified_correctly] = (guess_right[~classified_correctly] + guess_left[~classified_correctly]) / 2.0
        step += 1

    return adversarial_radii_, tolerances_


def get_adversarial_radii(
    model: nn.Module,
    loader: DataLoader,
    image_diap: Tuple[float, float],
    loss: nn.Module,
    max_steps: int = 100,
    tol: float = 1e-4,
    step_size: float = 0.01,
    device: str = 'cpu',
    norm: str = 'inf'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculating adversarial radii for all images in dataloader.
    :param model: pretrained network
    :param loader: data loader
    :param image_diap: min and max values of the images in loader, used for clipping
    :param loss: loss function
    :param max_steps: maximal number of algorithm steps (See get_adversarial_radii_on_batch())
    :param tol: maximum error of radii estimation to stop algorithm earlier
    (See get_adversarial_radii_on_batch())
    :param step_size: step_size parameter of algorithm (See get_adversarial_radii_on_batch())
    :param device: device type (cuda or cpu)
    :param norm: in what norm to make a step in the direction of the unitary vector,
    possible options 'inf'(classic fgsm, step to sign of gradient direction),
    'l2' (step in the gradient direction)
    :return: for each image: true label, adversarial radius, error of radius estimation,
     prediction of unattacked model
    """
    model.eval()
    req_grad(model, state=False)
    adversarial_radii = torch.tensor([], device=device)
    tolerances = torch.tensor([], device=device)

    targets = torch.tensor([], device=device)
    preds_unattacked = torch.tensor([], device=device)

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        images, labels = batch[0].to(device), batch[1].to(device)
        len_of_batch = labels.size()[0]
        targets = torch.cat([targets, labels.detach()])
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        preds_unattacked = torch.cat([preds_unattacked, preds.detach()])

        images_grad = get_image_grad_on_bacth(model, images, loss, norm)

        adversarial_radii_, tolerances_ = get_adversarial_radii_on_batch(
            model,
            len_of_batch,
            images,
            image_diap,
            preds,
            images_grad,
            max_steps=max_steps,
            step_size=step_size,
            tol=tol,
            device=device
        )
        adversarial_radii = torch.cat([adversarial_radii, adversarial_radii_])
        tolerances = torch.cat([tolerances, tolerances_])

    return (
        targets.cpu().numpy(),
        adversarial_radii.cpu().numpy(),
        tolerances.cpu().numpy(),
        preds_unattacked.cpu().numpy()
    )


def get_step_radii(
    model: nn.Module,
    loader: DataLoader,
    image_diap: Tuple[float, float],
    loss: nn.Module,
    eps: float,
    max_steps: int,
    device: str = 'cpu',
    norm: str = 'inf'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculating adversarial radii as the number of iterative attacks on the image.
    :param model: pretrained network
    :param loader: data loader
    :param image_diap: min and max values of the images in loader, used for clipping
    :param loss: loss function
    :param eps: attack step
    :param max_steps: maximal number of steps in iterative attack
    :param device: device type (cuda or cpu)
    :param norm: in what norm to make a step in the direction of the unitary vector,
    possible options 'inf' (classic fgsm, step to sign of gradient direction),
    'l2' (step in the gradient direction)
    :return: true labels, predictions of unattacked model, adversarial radii
    """
    model.eval()
    req_grad(model, state=False)

    targets = torch.tensor([], device=device)
    step_radii = torch.tensor([], device=device)
    preds_unattacked = torch.tensor([], device=device)

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        images, labels = batch[0].to(device), batch[1].to(device)
        targets = torch.cat([targets, labels.detach()])
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        preds_unattacked = torch.cat([preds_unattacked, preds])

        steps_radii_batch = torch.ones_like(labels, device=device) * np.inf
        class_is_unchanged = torch.ones_like(labels, dtype=torch.bool, device=device)
        step=0
        while True and (step < max_steps):
            preds_adv = preds.clone()
            images_grad = get_image_grad_on_bacth(model, images[class_is_unchanged],
                                                  loss, norm)
            images[class_is_unchanged], preds_adv[class_is_unchanged] = attack_step(model,
                                images[class_is_unchanged], images_grad, image_diap, eps)
            class_changed = preds_adv != preds
            steps_radii_batch[class_changed] = step
            class_is_unchanged[class_changed] = False
            if torch.all(~class_is_unchanged):
                break
            step+=1
        step_radii = torch.cat([step_radii, steps_radii_batch])

    return (
        targets.cpu().numpy(),
        preds_unattacked.cpu().numpy(),
        step_radii.cpu().numpy()
    )


def get_grad_norm(
    model: nn.Module,
    loader: DataLoader,
    loss: nn.Module,
    device: str = 'cpu',
    norm=2,
) -> np.ndarray:
    """Calculating norms of gradients for each image in dataloader.
    :param model: pretrained network
    :param loader: data loader
    :param loss: loss function
    :param device: device type (cuda or cpu)
    :param norm: which vector norm to use (the same as p parameter in torch.norm())
    :return: norms of gradients
    """
    model.eval()
    req_grad(model, state=False)
    grads_norms = torch.tensor([], device=device)

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        images = batch[0].to(device)
        images_grad = get_image_grad_on_bacth(model, images, loss, norm=None)
        grads_norms_batch = images_grad.reshape(images_grad.shape[0], -1).norm(p=norm, dim=1)
        grads_norms = torch.cat([grads_norms, grads_norms_batch])

    return grads_norms.cpu().numpy()


def get_adversarial_radii_on_ensemble(
    model_list: List[nn.Module],
    loader: DataLoader,
    image_diap: Tuple[float, float],
    loss: nn.Module,
    max_steps: int = 100,
    tol: float = 1e-4,
    step_size: float = 0.01,
    device: str = 'cpu',
    norm: str = 'inf',
    save: bool = True,
    path2save: str = 'default.npz'
):
    """Calculating adversarial radii for all images in dataloader.
    :param model: list of pretrained networks
    :param loader: data loader
    :param image_diap: min and max values of the images in loader, used for clipping
    :param loss: loss function
    :param max_steps: maximal number of algorithm steps (See get_adversarial_radii_on_batch())
    :param tol: maximum error of radii estimation to stop algorithm earlier
    (See get_adversarial_radii_on_batch())
    :param step_size: step_size parameter of algorithm (See get_adversarial_radii_on_batch())
    :param device: device type (cuda or cpu)
    :param norm: in what norm to make a step in the direction of the unitary vector,
    possible options 'inf'(classic fgsm, step to sign of gradient direction),
    'l2' (step in the gradient direction)
    :param save: if True, will save all outputs to .npz archive
    :param path2save: path in the format `.../filename.npz` to save adversarial radii for all base models
    (`adversaial_radii`), reduced versions(`mean_adversarial_radii`, `min_adversarial_radii`),
    targets (`targets`), predictions of unattacked model ('preds'),
    :return: (if save=False) for each image: true label, adversarial radius, error of radius estimation,
     prediction of unattacked model
    """
    adversarial_radii = None

    for model in model_list:
        targets, radii, tolerances, preds = get_adversarial_radii(
            model,
            loader,
            image_diap,
            loss,
            max_steps,
            tol,
            step_size,
            device,
            norm
        )
        if adversarial_radii is None:
            adversarial_radii = radii
        else:
            adversarial_radii = np.vstack([adversarial_radii, radii])

    if len(model_list) == 1:
      adversarial_radii = adversarial_radii[None, :]

    adversarial_radii_masked = np.ma.masked_array(adversarial_radii, np.isinf(adversarial_radii))
    adversarial_radii_masked_mean = adversarial_radii_masked.mean(axis=0)
    mean_adversarial_radii = adversarial_radii_masked_mean.data
    mean_adversarial_radii[adversarial_radii_masked_mean.mask] = np.inf

    adversarial_radii_masked_min = adversarial_radii_masked.min(axis=0)
    min_adversarial_radii = adversarial_radii_masked_min.data
    min_adversarial_radii[adversarial_radii_masked_min.mask] = np.inf

    if save:
        np.savez(path2save,
                 targets=targets,
                 preds=preds,
                 adversarial_radii=adversarial_radii,
                 min_adversarial_radii=min_adversarial_radii,
                 mean_adversarial_radii=mean_adversarial_radii
                 )
    else:
        return (targets,
                preds,
                adversarial_radii,
                min_adversarial_radii,
                mean_adversarial_radii
                )


def get_step_radii_on_ensemble(
    model_list: List[nn.Module],
    loader: DataLoader,
    image_diap: Tuple[float, float],
    loss: nn.Module,
    eps: float,
    max_steps: int,
    device: str = 'cpu',
    norm: str = 'inf',
    save: bool = True,
    path2save: str = 'default.npz'
):
    """Calculating adversarial radii as the number of iterative attacks on the image.
    :param model: list of pretrained networks
    :param loader: data loader
    :param image_diap: min and max values of the images in loader, used for clipping
    :param loss: loss function
    :param eps: attack step
    :param max_steps: maximal number of steps in iterative attack
    :param device: device type (cuda or cpu)
    :param norm: in what norm to make a step in the direction of the unitary vector,
    possible options 'inf'(classic fgsm, step to sign of gradient direction),
    'l2' (step in the gradient direction)
    :param save: if True, will save all outputs to .npz archive
    :param path2save: path in the format `.../filename.npz` to save step radii for all base models
    (`step_radii`), reduced versions(`mean_step_radii`, `min_step_radii`),
    targets (`targets`), predictions of unattacked model ('preds'),
    :return: (if save=False) true labels, predictions of unattacked model, adversarial radii
    """
    step_radii = None

    for model in model_list:
        targets, preds, radii = get_step_radii(
            model,
            loader,
            image_diap,
            loss,
            eps,
            max_steps,
            device,
            norm
        )
        if step_radii is None:
            step_radii = radii
        else:
            step_radii = np.vstack([step_radii, radii])

    if len(model_list) == 1:
      step_radii = step_radii[None, :]

    step_radii_masked = np.ma.masked_array(step_radii, np.isinf(step_radii))
    step_radii_masked_mean = step_radii_masked.mean(axis=0)
    mean_step_radii = step_radii_masked_mean.data
    mean_step_radii[step_radii_masked_mean.mask] = np.inf

    step_radii_masked_min = step_radii_masked.min(axis=0)
    min_step_radii = step_radii_masked_min.data
    min_step_radii[step_radii_masked_min.mask] = np.inf

    if save:
        np.savez(path2save,
                 targets=targets,
                 preds=preds,
                 adversaial_radii=step_radii,
                 min_step_radii=min_step_radii,
                 mean_step_radii=mean_step_radii
                 )
    else:
        return (targets,
                preds,
                step_radii,
                min_step_radii,
                mean_step_radii
                )


def get_maxprob_adversarial_radii(
    p2_p1: np.ndarray,
    model: nn.Module,
    loader: DataLoader,
    loss: nn.Module,
    device: str = 'cpu',
    norm=2,
    path2save='default.npz'
):
    """Calculate approximation of adversarial radius.
    :param p2_p1: difference between the highest and second highest probabilities
    from model predictions
    :param model: pretrained network
    :param loader: data loader
    :param loss: loss function
    :param device: device type (cuda or cpu)
    :param norm: which vector norm to use (the same as p parameter in torch.norm())
    :param path2save: path in the format `.../filename.npz` to save maxprob adversarial radii
    :return: approximated adversarial radii
    """
    grads_norms = get_grad_norm(
        model,
        loader,
        loss,
        device,
        norm
    )
    maxprob_adversarial_radii = grads_norms * p2_p1
    np.savez(path2save,
             mean_adversarial_radii=maxprob_adversarial_radii)
    return maxprob_adversarial_radii

# %%writefile utils/uncertainty_baselines.py
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import (
    List, Tuple, Dict, Any,
    Optional, Sequence, Callable
)
from torch.utils.data import DataLoader
from .attacks import req_grad

def entropy_from_logits(logits):
    return torch.sum(-F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)

def entropy_from_probas(probas):
    return torch.sum(-probas * torch.log(probas), dim=1)

def probas_std(probas):
    return torch.std(probas, dim=1)

def uncertainty_estimation(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cpu',
    save: bool=True,
    path2save: str = 'default.npz'
):
    """Calculate baseline uncertainty metrics.
    :param model: pretrained network
    :param loader: data loader
    :param device: device type (cuda or cpu)
    :param save: if True, will save all outputs to .npz archive
    :param path2save: path in the format `.../filename.npz`
    :return: (if save=False) true labels, predictions of unattacked model, entropy, (minus) standard
    deviation of predicted probas, 1 - maxprobs, (minus) difference between the highest and
    second highest probabilities

    Note: We take (minus) standard deviation of predicted probas, (minus) difference
    between the highest and second highest probabilities, (1 - maxprobs)
    to keep the same logic for all uncertainty metrics:
    the more is the metric the more is uncertainty.
    """
    model.eval()
    req_grad(model, state=False)

    targets = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    entropy_scores = torch.tensor([], device=device)
    std_scores = torch.tensor([], device=device)
    max_prob_scores = torch.tensor([], device=device)
    p1_p2_scores = torch.tensor([], device=device)

    for images, labels in tqdm(loader, total=len(loader)):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        preds_batch = torch.argmax(logits, dim=1)

        p = torch.topk(probas, k=2, dim=1).values
        max_prob = p[:, 0]
        p1_p2 = max_prob - p[:, 1]
        entropy = entropy_from_logits(logits)
        std = probas_std(probas)

        targets = torch.cat([targets, labels])
        preds = torch.cat([preds, preds_batch])
        entropy_scores = torch.cat([entropy_scores, entropy])
        std_scores = torch.cat([std_scores, std])
        max_prob_scores = torch.cat([max_prob_scores, max_prob])
        p1_p2_scores = torch.cat([p1_p2_scores, p1_p2])

    if save:
        np.savez(path2save,
                 targets=targets.cpu().numpy(),
                 preds=preds.cpu().numpy(),
                 entropy=entropy_scores.cpu().numpy(),
                 std=-std_scores.cpu().numpy(),
                 maxprobs=1-max_prob_scores.cpu().numpy(),
                 margin=-p1_p2_scores.cpu().numpy())
    else:
        return (targets.cpu().numpy(),
                preds.cpu().numpy(),
                entropy_scores.cpu().numpy(),
                -std_scores.cpu().numpy(),
                1 - max_prob_scores.cpu().numpy(),
                -p1_p2_scores.cpu().numpy())


def uncertainty_estimation_ensemble(
    model_list: List[nn.Module],
    loader: DataLoader,
    n_of_classes: int,
    save: bool=True,
    path2save: str = 'default.npz',
    device: str = 'cpu'
):
    """Calculate baseline uncertainty metrics for ensemble.
    :param model_list: list of pretrained networks
    :param loader: data loader
    :param n_of_classes: number of labels in classification task
    :param save: if True, will save all outputs to .npz archive
    :param path2save: path in the format `.../filename.npz`
    :param device: device type (cuda or cpu)
    :return: (if save=False) true labels, predictions of unattacked model,
    predicted entropy, expected entropy, mutual information, standard
    deviation of predicted probas for the most probable class across base models,
    (minus) mean standard deviation of predicted probas across base models, 1 - maxprobs

    Note: We take (minus) mean standard deviation of predicted probas across base models
    and (1 - maxprobs) to keep the same logic for all uncertainty metrics:
    the more is the metric the more is uncertainty.
    """

    for model in model_list:
        model.eval()
        req_grad(model, state=False)

    n_of_models = len(model_list)
    targets = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    predicted_entropy = torch.tensor([], device=device)
    expected_entropy = torch.tensor([], device=device)
    std_probas_modal_class = torch.tensor([], device=device)
    std_probas_mean = torch.tensor([], device=device)
    maxprobs = torch.tensor([], device=device)

    for images, labels in tqdm(loader, total=len(loader)):
        images, labels = images.to(device), labels.to(device)

        batch_size = labels.shape[0]
        logits_batch_ensemble = torch.zeros((batch_size, n_of_classes, n_of_models), device=device)
        for i, model in enumerate(model_list):
            logits_batch_ensemble[..., i] = model(images)

        probas_batch_ensemble = F.softmax(logits_batch_ensemble, dim=1)
        maxprobs_batch, preds_batch = probas_batch_ensemble.mean(dim=2).max(dim=1)
        expected_entropy_batch = entropy_from_logits(logits_batch_ensemble).mean(dim=1)
        predicted_entropy_batch = entropy_from_probas(probas_batch_ensemble.mean(dim=2))
        std_probas_modal_class_batch = torch.tensor([probas_batch_ensemble[j, preds_batch[j], :].std().item()
                                                    for j in range(batch_size)], device=device)
        std_probas_mean_batch = probas_std(probas_batch_ensemble).mean(dim=1)


        targets = torch.cat([targets, labels])
        preds = torch.cat([preds, preds_batch])
        predicted_entropy = torch.cat([predicted_entropy, predicted_entropy_batch])
        expected_entropy = torch.cat([expected_entropy, expected_entropy_batch])
        std_probas_modal_class = torch.cat([std_probas_modal_class, std_probas_modal_class_batch])
        std_probas_mean = torch.cat([std_probas_mean, std_probas_mean_batch])
        maxprobs = torch.cat([maxprobs, maxprobs_batch])

    mutual_information = predicted_entropy - expected_entropy

    if save:
        np.savez(path2save,
                 targets=targets.cpu().numpy(),
                 preds=preds.cpu().numpy(),
                 predicted_entropy=predicted_entropy.cpu().numpy(),
                 expected_entropy=expected_entropy.cpu().numpy(),
                 mutual_information=mutual_information.cpu().numpy(),
                 std_probas_modal_class=std_probas_modal_class.cpu().numpy(),
                 std_probas_mean=-std_probas_mean.cpu().numpy(),
                 maxprobs=1-maxprobs.cpu().numpy())
    else:
        return (targets.cpu().numpy(),
                preds.cpu().numpy(),
                predicted_entropy.cpu().numpy(),
                expected_entropy.cpu().numpy(),
                mutual_information.cpu().numpy(),
                std_probas_modal_class.cpu().numpy(),
                -std_probas_mean.cpu().numpy(),
                1 - maxprobs.cpu().numpy())


def get_best_vals_for_plot(preds, labels, rejection_rates):
    """Get best possible accs for each rejection rate

    :param preds: predictions
    :param labels: true labels
    :param rejection_rates: rejection_rates
    :return: rejection rates and best accuracies
    """
    preds_total = len(labels)
    preds_correct = (preds == labels).sum()
    preds_incorrect = (preds != labels).sum()

    accs = []
    for rate in rejection_rates:
        num_of_rejected = min(np.floor(preds_total * rate), preds_incorrect)
        accs.append(preds_correct / (preds_total - num_of_rejected))

    return accs


def sort_data_by_metric(
    metric: np.ndarray, preds: np.ndarray, labels: np.ndarray
) -> Tuple[List, List]:
    """Sort preds and labels by descending uncertainty metric.
    :param metric: uncertainty metric according to which preds and labels will be sorted
                   (Expected logic: The more is uncertainty metric, the more is uncertainty)
    :param preds: model predictions
    :param labels: ground truth labels
    :return: a tuple of
        - np.ndarray of predictions, sorted according to metric
        - np.ndarray of labels, sorted according to metric
    """
    sorted_metric_idx = np.argsort(metric)
    return preds[sorted_metric_idx], labels[sorted_metric_idx]


def reject_by_metric(
    uncertainty_proxy: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    rejection_rates: np.ndarray,
    scoring_func: Callable,
) -> List:
    """Reject points from preds and labels based on uncertainty estimate of choice.
    :param uncertainty_proxy: array of unceratinty estimations (e.g -adversarial radii)
    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param rejection_rates: list of rejection rates to use
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each rejection_rate
    """
    preds_sorted, labels_sorted = sort_data_by_metric(uncertainty_proxy, preds, labels)
    upper_bounds = len(preds) - np.ceil(rejection_rates * len(preds))

    scores = []
    for upper_bound in upper_bounds[:-1].astype('int'):
        scores.append(scoring_func(labels_sorted[:upper_bound], preds_sorted[:upper_bound]))

    return scores

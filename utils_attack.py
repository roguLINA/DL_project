from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Tuple, Dict, Any, Optional, Sequence, Callable

def calculate_metrics(
    target: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """Calculate Accuracy, ROC_AUC, PR_AUC and confusion matrix.

    :param target: array with true labels
    :param y_pred: array with predictions
    :return: tuple of
        - Accuracy
        - ROC_AUC
        - PR_AUC
        - Confusion matrix (TN, FP, FN, TP)
    """
    acc = (target == y_pred).mean()
#     roc_auc = roc_auc_score(target, y_pred, multi_class='ovo')
#     pr_auc = average_precision_score(target, y_pred, average='macro')

    return acc #, roc_auc, pr_auc

def req_grad(model: nn.Module, state: bool = True) -> None:
    """Set requires_grad of all model parameters to the desired value.

    :param model: the model
    :param state: desired value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad_(state)

def test_on_adv(
    model: nn.Module,
    loader: DataLoader,
    loss: nn.Module,
    params: Dict[str, Any],
    method: str = "fgsm",
    n_samples_ret: int = 5,
    device: str = 'cpu'
    #use_preds: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create adversarial inputs and test model on it.

    :param model: model that will be evaluated
    :param loader: DataLoader from DatasetSlices
    :param loss: loss function
    :param params: attack params, for FGSM the only parameter is 'eps', for PGD also 'n_steps'
    :param method: attack method, one of 'fgsm' and 'pgd'
    :param n_samples_ret: how many batches of samples should be returned
    :param use_preds: if True, change of class if tracked between adversarial sample class and predicted class; if False,
        change of class is tracked between adversarial sample class and ground truth class
    :returns: tuple of
        - y_true
        - y_pred adversarial
        - y_pred
        - adversarial samples, n_samples_ret batches
        - adversarial targets
        - ground truth samples, n_samples_ret batches
        - ground truth targets
    """
    model.eval()
    loss_hist = []
    acc_hist = []

    loss_adv_hist = []
    acc_adv_hist = []

    req_grad(model, state=False)  # detach all model's parameters

    targets = torch.tensor([], device=device)
    total_preds = torch.tensor([], device=device)
    ori_preds = torch.tensor([], device=device)
    adv_samples = torch.tensor([], device=device)
    adv_targets = torch.tensor([], device=device)
    true_samples = torch.tensor([], device=device)
    true_targets = torch.tensor([], device=device)

    for x_ori, labels in loader:
        x_ori, labels = x_ori.to(device), labels.to(device)
        x_adv = torch.clone(x_ori)
        x_adv.requires_grad = True

        # prediction for original input
        logits = model(x_adv)
        _, preds = torch.max(logits, dim=1)
#         preds_th = (preds.view(-1,) > 0.5).float().detach()
#         targets.extend(preds_th if use_preds else labels.detach().data.numpy())
        targets = torch.cat([targets, labels])
        ori_preds = torch.cat([ori_preds, preds])
#         loss_val = loss(logits.view(-1,), preds_th if use_preds else labels.float(),)
        loss_val = loss(logits, labels)
        loss_val.backward()

        x_adv.data = x_adv.data + params["eps"] * torch.sign(x_adv.grad.data)

        # perturbations
        if method == "fgsm":
            steps = 0
            noise = params["eps"] * torch.sign(x_adv.grad.data)
        elif method == "pgd":
            steps = params["steps"] - 1
            x_adv.data = torch.max(
                x_ori - params["eps"], torch.min(x_adv.data, x_ori + params["eps"])
            )

        logits_adv = model(x_adv)
#         loss_adv = loss(logits_adv.view(-1,), preds_th if use_preds else labels.float(),)
        loss_adv = loss(logits_adv, labels)
        loss_adv.backward()

        for k in range(steps):
            x_adv.data = x_adv.data + params["alpha"] * torch.sign(x_adv.grad.data)
            x_adv.data = torch.max(
                x_ori - params["eps"], torch.min(x_adv.data, x_ori + params["eps"])
            )

            logits_adv = model(x_adv)
#             loss_adv = loss(logits_adv.view(-1,), preds_th if use_preds else labels.float(),)
            loss_adv = loss(logits_adv, labels)
            loss_adv.backward()

        # predictions for adversarials
        _, preds_adv = torch.max(logits_adv, dim=1)
        total_preds = torch.cat([total_preds, preds_adv])

        # accuracy
        acc_val = (preds == labels).float().sum() / len(labels)
        acc_adv = (preds_adv == labels).float().sum() / len(labels)
#         acc_val = np.mean((preds == logits if use_preds else labels).numpy())
#         acc_adv = np.mean((preds_adv == logits if use_preds else labels).numpy())

        loss_hist.append(loss_val.cpu().item())
        acc_hist.append(acc_val)

        loss_adv_hist.append(loss_adv.cpu().item())
        acc_adv_hist.append(acc_adv)

        if np.random.rand() > 0.8 and len(adv_samples) < n_samples_ret:
            adv_samples = torch.cat([adv_samples, x_adv])
            adv_targets = torch.cat([adv_targets, preds_adv])
            true_samples = torch.cat([true_samples, x_ori])
            true_targets = torch.cat([true_targets, preds])

    return (
        targets, 
        total_preds, 
        ori_preds, 
        adv_samples,
        adv_targets, 
        true_samples, 
        true_targets,
    )

def test_robustness_simple(
    model,
    test_loader,
    criterion,
#     params={"eps": 1e-3},
    attack_params: Tuple[float, float, int],
    attack_type: str = "fgsm",
    n_samples_ret: int = 5,
    device: str = 'cpu',
#     heteroscedastic: bool = False,
) -> pd.DataFrame:
    """Test model robustness with homogeneous noise or FGSM attack.

    :param model: tne model to evaluate
    :param df: the data
    :param selected_feature: the features that the model uses
    :param train_wells: indices of training wells, they will not be used for evaluation
    :param test_wells: indices of testing wells, they will be used for evaluation
    :param slice_len: length of slice that the model uses
    :param path_to_saves: path to folder with scaler save
    :param attack_type: one of 'noise' and 'adversarial'
    :param attack_params: lower bound, upper bound and number of points for noise sigma or attack power
    :param heteroscedastic: if True, the noise sigma depends on the value of the input
    :returns: DataFrame with results
    """
    results = dict()
    for eps in tqdm(np.geomspace(*attack_params)):
        y_true, y_pred, _, _, _, _, _ = test_on_adv(
            model,
            test_loader,
            loss=criterion,
            params={"eps": eps},
            method=attack_type,
            n_samples_ret=n_samples_ret,
            device=device
        )
        results[eps] = [calculate_metrics(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())]
#         results[eps / 3] = calculate_metrics(y_true, y_pred)

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.set_axis(
        pd.Index(["Accuracy"], name="Metric"), axis=1, inplace=True
    )
#     results_df.set_axis(
#         pd.Index(["Accuracy", "ROC AUC", "PR AUC"], name="Metric"), axis=1, inplace=True
#     )
    results_df.set_axis(
        pd.Index(
            results_df.index,
            name="Noise sigma",
        ),
        axis=0,
        inplace=True,
    )

    return results_df

def _adversarial_radius(
    model: nn.Module,
    dataloader: DataLoader,
    loss: nn.Module,
    eps: float,
    n_steps: int,
    method: str,
    results: np.ndarray,
    device: str = 'cpu',
    #use_preds: bool
) -> np.ndarray:
    """Evaluate change of class of samples under adversarial attack.

    :param model: the model to evaluate
    :param dataloader: the data
    :param eps: attack power
    :param n_steps: number of steps
    :param method: attack type
    :param results: current results
    :param use_preds: if True, change of class if tracked between adversarial sample class and predicted class; if False,
        change of class is tracked between adversarial sample class and ground truth class
    :returns: indices of samples that changed class
    """
    y_true, y_adv, y_pred, _, _, _, _ = test_on_adv(
        model,
        loader=dataloader,
        loss=loss,
        params={"eps": eps, "steps": n_steps, "alpha": eps},
        method=method,
        device=device,
        #use_preds=use_preds
    )
    y_true = y_true.detach().cpu().numpy()
    y_adv = y_adv.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

#     y_true = (y_true>0.5).astype(float)
#     y_pred = (y_pred>0.5).astype(float)
#     y_adv = (y_adv>0.5).astype(float)

#     if use_preds:
#         update = np.flatnonzero(y_pred.flatten() != y_adv.flatten())
#     else:
#         update = np.flatnonzero(y_true.flatten() != y_adv.flatten())

    update = np.flatnonzero(y_pred != y_adv)

    nnz = np.flatnonzero(results)
    update = np.array(list(set(update) - set(nnz)), dtype='int')

    return update


def adversarial_radius(
    model: nn.Module,
    dataloader: DataLoader,
    loss: nn.Module,
    num_samples: int,
    min_v: float,
    max_v: Optional[float],
    step_v: float,
    method: str,
    device: str = 'cpu',
    #use_preds: bool,
) -> np.ndarray:
    """Evaluate adversarial radius of samples.

    :param model: the model to evaluate
    :param dataloader: the data
    :param num_samples: number of samples
    :param min_v: minimal attack power for 'fgsm' method, minimal number of steps for 'pgd' method
    :param max_v: maximal attack power for 'fgsm' method, maximal number of steps for 'pgd' method; if not given,
        no limit is imposed
    :param step_v: multiplicative step in attack power for 'fgsm' method or additive step in number of steps
        for 'pgd' method
    :param method: attack type
    :param use_preds: if True, change of class if tracked between adversarial sample class and predicted class; if False,
        change of class is tracked between adversarial sample class and ground truth class
    :returns: np.ndarray with minimal attack eps that changes class
    """
    results = np.zeros(num_samples)
    
    if max_v is not None:
        if method == "fgsm":
            num_v = int((np.log10(max_v) - np.log10(min_v)) / np.log10(step_v)) + 1
            iterable = np.geomspace(min_v, max_v, num_v)
        else:
            num_v = int((max_v - min_v) / step_v) + 1
            iterable = np.linspace(min_v, max_v, num_v)
        for v in tqdm(iterable):
            update = _adversarial_radius(
                model,
                dataloader,
                loss,
                v if method == "fgsm" else 1e-3,
                int(v) if method == "pgd" else 1,
                method,
                results,
                device=device,
                #use_preds
            )
#             print('update', update.shape)
#             print('results', results.shape)
            if len(update) > 0:
                results[update] = v
        results[results == 0] = np.inf
    else:
        num_nonzero = len(np.flatnonzero(results))
        v = min_v
        while num_nonzero < num_samples:
            update = _adversarial_radius(
                model,
                dataloader,
                loss,
                v if method == "fgsm" else 1e-3,
                int(v) if method == "pgd" else 1,
                method,
                results,
                device=device,
                #use_preds
            )
            if len(update) > 0:
                results[update] = v
            num_nonzero = num_nonzero - len(update)
            if method == "fgsm":
                v = v * step_v
            else:
                v = v + step_v

    return results

def get_adversarial_radii(
    model: nn.Module, # was list of models (because ensembles were used)
    preds: np.ndarray,
    dataloader: DataLoader,
    loss: nn.Module,
    num_samples: int,
    min_v: float,
    max_v: Optional[float],
    step_v: float,
    method: str,
    device: str = 'cpu',
    #use_preds: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param model: model to evaluate 
    :param preds: numpy array of ensemble predictions. Expects first dimension to represent members of ensemble
    :param dataloader: the data
    :param num_samples: number of samples
    :param min_v: minimal attack power for 'fgsm' method, minimal number of steps for 'pgd' method
    :param max_v: maximal attack power for 'fgsm' method, maximal number of steps for 'pgd' method; if not given,
        no limit is imposed
    :param step_v: multiplicative step in attack power for 'fgsm' method or additive step in number of steps
        for 'pgd' method
    :param method: attack type
    :param use_preds: if True, change of class if tracked between adversarial sample class and predicted class; if False,
        change of class is tracked between adversarial sample class and ground truth class
    :returns: np.ndarray with average predictions, np.ndarray with minimal attack eps that changes class
    """
    ave_preds = np.average(np.copy(preds), axis=0)
    
    func = partial(
        adversarial_radius,
        dataloader=dataloader,
        loss=loss,
        num_samples=num_samples,
        min_v=min_v,
        max_v=max_v,
        step_v=step_v,
        method=method,
        device=device,
        #use_preds=use_preds,
    )

    radii = [func(model)] #[func(m) for m in models]

    def _finite_mean(arr_1d):
        mask = np.isfinite(arr_1d)
        if mask.sum() == 0:
            return np.inf
        else:
            return np.mean(arr_1d[mask])

    return ave_preds, np.apply_along_axis(_finite_mean, 0, radii)











def sort_data_by_metric(
    metric: Sequence, preds: np.ndarray, labels: np.ndarray
) -> Tuple[List, List]:
    """Sort preds and labels by descending uncertainty metric.

    :param metric: uncertainty metric according to which preds and labels will be sorted
    :param preds: model predictions
    :param labels: ground truth labels
    :return: a tuple of
        - np.ndarray of predictions, sorted according to metric
        - np.ndarray of labels, sorted according to metric
    """
    sorted_metric_idx = np.argsort(metric)

    return preds[sorted_metric_idx], labels[sorted_metric_idx]

def get_upper_bound_idx(data_len: int, rejection_rates: Sequence[float]) -> List[float]:
    """Calculate upped bounds on indices of data arrays.

    Based on corresponding list of rejection rates is applied.

    :param data_len: length of data array
    :param rejection_rates: array of rejection rates to calculate upper bounds for
    :return: list of upper bounds
    """
    idx = []
    for rate in rejection_rates:
        idx.append(
            min(np.ceil(data_len * (1 - rate)), np.array(data_len)).astype(int).item()
        )

    return idx

def reject_and_eval(
    preds: np.ndarray,
    labels: np.ndarray,
    upper_bounds: Sequence[float],
    scoring_func: Callable,
) -> List:
    """Clip preds and labels arrays.

    Using list of upper bounds, and calculate scoring metric for
    predictions after rejection.

    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param upper_bounds: list of upper bounds to clip preds and labels to
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each upper bound
    """
    scores = []
    #predicted_labels = np.where(preds > 0.5, 1, 0)

    for upper_bound in upper_bounds:
        #predicted_labels_below_thresh = predicted_labels[0:upper_bound]
        predicted_labels_below_thresh = preds[0:upper_bound]
        preds_below_thresh = preds[0:upper_bound]
        labels_below_thresh = labels[0:upper_bound]
        try:
            if preds_below_thresh.size > 0 and labels_below_thresh.mean() not in [0, 1]:
                scores.append(scoring_func(labels_below_thresh, preds_below_thresh))
        except ValueError:
            if (
                predicted_labels_below_thresh.size > 0
                and labels_below_thresh.mean() not in [0, 1]
            ):
                scores.append(
                    scoring_func(labels_below_thresh, predicted_labels_below_thresh)
                )

    return scores

def reject_by_metric(
    get_metric: Callable,
    preds: np.ndarray,
    labels: np.ndarray,
    rejection_rates: List[float],
    scoring_func: Callable,
) -> List:
    """Reject points from preds and labels based on uncertainty estimate of choice.

    :param get_metric: function that returns uncertainty metric for given model predictions
    :param preds: model label predictions or predicted class probabilities
    :param labels: ground truth labels
    :param rejection_rates: list of rejection rates to use
    :param scoring_func: scoring function that takes labels and predictions or probabilities (in that order)
    :return: list of scores calculated for each upper bound
    """
    preds, metric_values = get_metric(preds)
    preds_sorted, labels_sorted = sort_data_by_metric(metric_values, preds, labels)

    upper_indices = get_upper_bound_idx(preds.size, rejection_rates)
    return reject_and_eval(preds_sorted, labels_sorted, upper_indices, scoring_func)
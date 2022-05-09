from tqdm import tqdm
import numpy as np
import pandas as pd 

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Tuple, Dict, Any

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
#         x_ori = torch.stack(x, dim=0)
        x_adv = torch.clone(x_ori)
        x_adv.requires_grad = True

        # prediction for original input
        logits = model(x_adv)
#         preds = logits.detach()
        _, preds = torch.max(logits, dim=1)
#         preds_th = (preds.view(-1,) > 0.5).float().detach()
#         targets.extend(preds_th if use_preds else labels.detach().data.numpy())
        targets = torch.cat([targets, labels])
        ori_preds = torch.cat([ori_preds, preds])
#         targets.extend(labels.detach().cpu().data.numpy())
#         ori_preds.extend(preds.detach().cpu().data.numpy())
#         loss_val = loss(logits.view(-1,), preds_th if use_preds else labels.float(),)
#         loss_val = loss(logits.view(-1,), labels.float())
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
#         _, preds_adv = torch.max(logits_adv, dim=1)
#         loss_adv = loss(logits_adv.view(-1,), preds_th if use_preds else labels.float(),)
#         loss_adv = loss(logits_adv.view(-1,), labels.float())
        loss_adv = loss(logits_adv, labels)
        loss_adv.backward()

        for k in range(steps):
            x_adv.data = x_adv.data + params["alpha"] * torch.sign(x_adv.grad.data)
            x_adv.data = torch.max(
                x_ori - params["eps"], torch.min(x_adv.data, x_ori + params["eps"])
            )

            logits_adv = model(x_adv)
#             _, preds_adv = torch.max(logits_adv, dim=1)
#             loss_adv = loss(logits_adv.view(-1,), preds_th if use_preds else labels.float(),)
#             loss_adv = loss(logits_adv.view(-1,), labels.float())
            loss_adv = loss(logits_adv, labels)
            loss_adv.backward()

        # predictions for adversarials
#         preds_adv = logits_adv.detach()
        _, preds_adv = torch.max(logits_adv, dim=1)
#         total_preds.extend(preds_adv.detach().cpu().data.numpy())
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
#             adv_samples.append(x_adv.data.numpy())
#             adv_targets.append(preds_adv.data.numpy())
#             true_samples.append(x_ori.data.numpy())
#             true_targets.append(preds.data.numpy())

    return (
        targets, #np.array(targets),
        total_preds, #np.array(total_preds),
        ori_preds, #np.array(ori_preds),
        adv_samples, #np.array(adv_samples),
        adv_targets, #np.array(adv_targets),
        true_samples, #np.array(true_samples),
        true_targets, #np.array(true_targets),
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
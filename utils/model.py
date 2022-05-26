# %%writefile utils/model.py
import torch
from tqdm import tqdm


def epoch_train(loader, clf, criterion, opt, device='cpu'):
    losses, accuracies = 0, 0

    clf.train(True)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        output = clf(data)
        loss = criterion(output, target)

        _, output_tags = torch.max(output, dim=1)
        acc = (output_tags == target).float().sum() / len(target)

        losses += loss.item()
        accuracies += acc.data.item()

        loss.backward()
        opt.step()

    return losses / (batch_idx + 1), accuracies / (batch_idx + 1), clf

def epoch_test(loader, clf, criterion, device='cpu'):
    losses, accuracies = 0, 0

    with torch.no_grad():
        clf.eval()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            prediction = clf(data)
            loss = criterion(prediction, target)

            _, pred_tags = torch.max(prediction, dim=1)
            acc = (pred_tags == target).float().sum() / len(target)

            losses += loss
            accuracies += acc.data.item()

    return losses / (batch_idx + 1), accuracies / (batch_idx + 1)

def train(train_loader, test_loader, clf, criterion, opt, sch, n_epochs=50, device='cpu'):
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc, clf = epoch_train(train_loader, clf, criterion, opt, device)
        test_loss, test_acc = epoch_test(test_loader, clf, criterion, device)

        print(f'[Epoch {epoch + 1}] train loss: {train_loss:.3f}; train acc: {train_acc:.2f}; ' +
              f'test loss: {test_loss:.3f}; test acc: {test_acc:.2f}')
        sch.step()

    return clf

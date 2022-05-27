# %%writefile utils_ensembles.py
import torch
from torch import nn
from copy import deepcopy
import utils_model
import utils_data
import numpy as np


class EnsembleModel:
    def __init__(self,
                 n_of_models,
                 bagging,
                 model,
                 n_epochs,
                 device,
                 dataset_type,
                 root_data_folder,
                 batch_size=64):
        super().__init__()

        self.n_of_models = n_of_models
        self.bagging = bagging
        self.base_model = model
        self.n_epochs = n_epochs
        self.device = device
        self.dataset_type = dataset_type
        self.root_data_folder = root_data_folder
        self.batch_size = batch_size
        self.trained_model_list = []

    def fine_tune_model(self, i):
        utils_data.set_random_seeds(seed_value=i, device=self.device)

        model = deepcopy(self.base_model).to(self.device)
        criterion = nn.CrossEntropyLoss()

        '''source https://github.com/kuangliu/pytorch-cifar/blob/master/main.py'''
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                                    momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


        (train_set, test_set), (train_loader, test_loader) = utils_data.get_train_test_dataloaders(
            dataset_type=self.dataset_type,
            root_data_folder=self.root_data_folder,
            batch_size=self.batch_size
        )
        
        if self.bagging:
            train_indices = np.random.choice(range(50000), size=25000, replace=False)
            test_indices = np.random.choice(range(10000), size=5000, replace=False)
            train_subset, test_subset = torch.utils.data.Subset(train_set, train_indices), torch.utils.data.Subset(test_set, test_indices)
            
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=self.batch_size,
                shuffle=True, num_workers=2
            )
            test_loader = torch.utils.data.DataLoader(
                test_subset, batch_size=self.batch_size,
                shuffle=False, num_workers=2
            )

        model = utils_model.train(
            train_loader, test_loader, model,
            criterion, optimizer, scheduler,
            self.n_epochs, device=self.device
        )
        path2save = f'models/{self.dataset_type}/model_{i}.pt'
        self.save_model(model, path2save)
        del model


    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def train(self):
        for i in range(self.n_of_models):
            self.fine_tune_model(i)

    def get_model(self, i):
        model = deepcopy(self.base_model).to(self.device)
        path = f'models/{self.dataset_type}/model_{i}.pt'
        model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        return model

    def get_all_models(self):
        for i in range(self.n_of_models):
            self.trained_model_list.append(self.get_model(i))


#USAGE:
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# model_10 = torchvision.models.resnet50(pretrained=True)
# model_10.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
# model_10 = model_10.to(device)
# ensemble_10.get_all_models()


# ensemble_10 = utils_ensembles.EnsembleModel(
#     n_of_models=5,
#     bagging=False,
#     model=model_10,
#     n_epochs=1,
#     device=device,
#     dataset_type='cifar10',
#     root_data_folder='./data',
#     batch_size=64)
#
# ensemble_10.train()
# ensemble_10.get_all_models()

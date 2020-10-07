import sys
import os
import copy
import logging
from time import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocess import preprocess
from models.models import CustomLinear


from models.utils import save_feature_impotances, save_params


class NeuralNet:

    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.fold_type = fold_type
        self.n_splits = n_splits
        self.batch_size = 32
        self.train_epochs = 5
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')

    def trainer(self):
        print(self.X.shape, self.X_test.shape)
        X = self.X.values
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = self.y.values
        test = self.X_test.values
        scaler = StandardScaler()
        test = scaler.fit_transform(test)
        test = torch.from_numpy(test.astype(np.float32))
        test_dataset = torch.utils.data.TensorDataset(test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        net = nn.Sequential(CustomLinear(X.shape[1], 2))

        test_pred_all = np.zeros(len(test))

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        for i, (train_idx, valid_idx) in enumerate(kf.split(X)):

            x_train_fold = torch.from_numpy(X[train_idx].astype(np.float32))
            y_train_fold = torch.from_numpy(
                y[train_idx, np.newaxis].astype(np.float32))
            x_val_fold = torch.from_numpy(X[valid_idx].astype(np.float32))
            y_val_fold = torch.from_numpy(
                y[valid_idx, np.newaxis].astype(np.float32))

            model = net.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)
            criterion = nn.CrossEntropyLoss()

            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=self.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.batch_size, shuffle=True)

            print('----------')
            print('Start fold {}/{}'.format(i, self.n_splits))

            # epoch分のループを回す
            for epoch in range(self.train_epochs):
                model.train()
                avg_loss = 0
                y_preds_list = []
                y_batch_list = []

                for x_batch, y_batch in train_loader:
                    preds = model(x_batch)
                    y_batch = y_batch.flatten().long()
                    loss = criterion(preds, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # accuracyの計算
                    y_preds = preds.detach().numpy().copy()
                    y_preds = np.array([i[1] for i in y_preds])
                    y_batch = y_batch.detach().numpy().copy()
                    # 足す
                    y_preds_list.extend(y_preds)
                    y_batch_list.extend(y_batch)
                    avg_loss += loss.item() / len(train_loader)

                accuracy = self.metric(y_batch_list, y_preds_list)
                scheduler.step()
                model.eval()
                avg_val_loss = 0
                y_val_preds_list = []
                y_val_batch_list = []

                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    preds = model(x_batch)
                    y_batch = y_batch.flatten()
                    loss = criterion(preds, y_batch.long())
                    y_preds = preds.detach().numpy().copy()
                    y_preds = np.array([i[1] for i in y_preds])
                    y_batch = y_batch.detach().numpy().copy()
                    # 足す
                    y_val_preds_list.extend(y_preds)
                    y_val_batch_list.extend(y_batch)
                    avg_val_loss += loss.item() / len(valid_loader)

                val_accuracy = self.metric(y_val_batch_list, y_val_preds_list)

                print('Epoch {}/{} \t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f} \t val_accuracy={:.4f} '.format(
                    epoch + 1, self.train_epochs, avg_loss, accuracy, avg_val_loss, val_accuracy))

            test_pred = np.array([])
            test_dataset = torch.utils.data.TensorDataset(test)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False)

            # X_test_fold をbatch_sizeずつ渡すループ
            for i, (x_batch, ) in enumerate(test_loader):
                y_pred = model(x_batch)
                y_pred = y_pred.detach().numpy().copy()
                y_pred = np.array([i[1] for i in y_pred])
                test_pred = np.append(test_pred, y_pred)
            test_pred_all += test_pred

        test_pred_all = test_pred_all / self.n_splits
        return test_pred_all

    def metric(self, va_y, pred):
        pred = list(pred)
        va_y = list(va_y)
        score = average_precision_score(va_y, pred)
        if str(score) == 'nan':
            print(va_y, pred)
            sys.exit()
        return score

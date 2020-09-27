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
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from models.utils import save_feature_impotances, save_params


class lightGBM:

    def __init__(self, X, y, X_test, output_path, fold_type, n_slits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            # 'metric': 'auc',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'max_depth': -1,
            'num_leaves': 32,
            'max_bin': 32,
            'nthread': -1,
            'bagging_freq': 1,
            'verbose': -1,
            'seed': 2020,
        }
        self.fold_type = fold_type
        self.n_splits = n_slits

    def trainer(self):
        y_pred = np.zeros(len(self.X_test))
        scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        if self.fold_type == 'kfold':
            for i, (tr_idx, va_idx) in enumerate(kf.split(self.X)):
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

                score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)
                
                y_pred += pred / self.n_splits

            final_score = sum(scores) / self.n_splits

        elif self.fold_type == 'oof':
            tr_x, va_x, tr_y, va_y = train_test_split(self.X, self.y, test_size=0.2)

            score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
            scores.append(score)
            self.epoch_log(score, 0, 1)
            y_pred += pred
            final_score = score

        print('avarage_accuracy: {}'.format(final_score))
        logging.info('avarage_accuracy: {}'.format(final_score))

        feature_importances = model.feature_importance()
        save_feature_impotances(
            self.X.columns, feature_importances, self.output_path)
        save_params(self.params, self.output_path)
        return y_pred

    def train(self, tr_x, tr_y, va_x, va_y):
        train_data = lgb.Dataset(tr_x, tr_y)
        valid_data = lgb.Dataset(va_x, va_y)

        model = lgb.train(self.params, train_data, valid_sets=[train_data, valid_data],
                          num_boost_round=10000, early_stopping_rounds=200,
                          verbose_eval=200)

        y_val_pred = model.predict(va_x)
        score = self.metric(va_y, y_val_pred)
        score = score

        pred = model.predict(self.X_test)

        return score, pred, model

    def epoch_log(self, score, i, n_splits):
        print('{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))
        logging.info('{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))

    def metric(self, va_y, pred):
        return np.sqrt(mean_squared_error(va_y, pred))
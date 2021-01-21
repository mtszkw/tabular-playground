import json
import logging

import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn

from metrics import rmse


class TabNetOptunaObjective(object):
    def __init__(self, df, feature_col, target_col, random_state, default_params):
        self.df = df
        self.feature_col = feature_col
        self.target_col = target_col
        self.random_state = random_state
        self.default_params = default_params


    def __call__(self, trial):
        df_train, df_valid = train_test_split(self.df, test_size=0.1, random_state=self.random_state)
        X_train, y_train = df_train[self.feature_col].values, df_train[self.target_col].values.reshape(-1, 1)
        X_valid, y_valid = df_valid[self.feature_col].values, df_valid[self.target_col].values.reshape(-1, 1)
        logging.info(f'Train/valid split: {X_train.shape[0]} for training, {X_valid.shape[0]} for validation')
        
        n_d = trial.suggest_int('n_d', 8, 64)
        max_lr = trial.suggest_float('max_lr', 1e-4, 5e-2)

        default_params['n_d'] = n_d
        default_params['n_a'] = n_d
        default_params['n_steps'] = trial.suggest_int('n_steps', 3, 10)
        default_params['gamma'] = trial.suggest_float('gamma', 1.0, 2.0)
        default_params['optimizer_params'] = optimizer_params=dict(lr=max_lr, weight_decay=1e-5)
        default_params['scheduler_params'] = dict(base_lr=1e-6, max_lr=max_lr, cycle_momentum=False)

        model = TabNetRegressor(**default_params)

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['rmse'],
            max_epochs=20,
            patience=20,
            batch_size=1024)

        score = rmse(y_valid, model.predict(X_valid).squeeze())
        return score


class TabNetTrainer:
    def __init__(self, random_state: int):
        self.random_state = random_state

    def default_params():
        return {
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=max_lr, weight_decay=1e-5),
            'scheduler_fn': torch.optim.lr_scheduler.CyclicLR,
            'scheduler_params': dict(base_lr=1e-6, max_lr=max_lr, cycle_momentum=False),
            'device_name': 'auto'}

    def optimize_hyperparameters(self, df: pd.DataFrame, feature_col: list, target_col: str, output_file: str) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(TabNetOptunaObjective(df, feature_col, target_col, self.random_state, self.default_params()), n_trials=25)
        
        logging.info(f"Best Optuna trial for TabNet: {study.best_trial.value}")
        logging.info(f'Best TabNet parameters found: {study.best_trial.params}')

        with open(output_file, "w") as f:
            f.write(json.dumps(study.best_trial.params))
        return study.best_trial.params


    def crossval_and_predict(self, n_folds: int, df: pd.DataFrame, df_test: pd.DataFrame, feature_col: list, target_col: str, model_params: dict):
        oof = np.zeros((len(df)))
        cv_preds = np.zeros((len(df_test)))
        kfold = KFold(n_splits=n_folds, random_state=self.random_state, shuffle=True)
        for train_idx, valid_idx in kfold.split(df):
            X_train, y_train = df[feature_col].values[train_idx], df[target_col].values[train_idx].reshape(-1, 1)
            X_valid, y_valid = df[feature_col].values[valid_idx], df[target_col].values[valid_idx].reshape(-1, 1)
            X_test = df_test[feature_col].values

            params = self.default_params()
            n_d = trial.suggest_int('n_d', 8, 64)
            max_lr = trial.suggest_float('max_lr', 1e-4, 5e-2)

            params['n_d'] = model_params['n_d']
            params['n_a'] = model_params['n_d']
            params['n_steps'] = model_params['momentum']
            params['gamma'] = model_params['gamma']
            params['optimizer_params'] = optimizer_params=dict(lr=model_params['max_lr'], weight_decay=1e-5)
            params['scheduler_params'] = dict(base_lr=1e-6, max_lr=model_params['max_lr'], cycle_momentum=False)


            model = TabNetRegressor(**model_params)
            model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=['rmse'],
                max_epochs=100,
                patience=50,
                batch_size=1024)

            oof[valid_idx] = model.predict(X_valid).squeeze()
            cv_preds += model.predict(X_test).squeeze() / n_folds
        
        rmse_score = rmse(df[target_col], oof)
        return rmse_score, cv_preds

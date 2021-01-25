import json
import logging

import numpy as np
import optuna
import pandas as pd
from pytorch_tabnet.pretraining import  TabNetPretrainer
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

        params = self.default_params
        params['n_d'] = n_d
        params['n_a'] = n_d
        params['seed'] = self.random_state
        params['n_steps'] = trial.suggest_int('n_steps', 3, 10)
        params['n_shared'] = trial.suggest_int('n_shared', 2, 5)
        params['n_independent'] = trial.suggest_int('n_independent', 2, 5)
        params['momentum'] = trial.suggest_float('momentum', 0.01, 0.4)
        params['gamma'] = trial.suggest_float('gamma', 1.0, 2.0)

        model = TabNetRegressor(**params)

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['rmse'],
            max_epochs=20,
            patience=10,
            batch_size=1024)

        score = rmse(y_valid, model.predict(X_valid).squeeze())
        return score


class TabNetTrainer:
    def __init__(self, random_state: int):
        self.random_state = random_state

    def default_params(self):
        return {
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
            'scheduler_fn': torch.optim.lr_scheduler.StepLR,
            'scheduler_params': dict(step_size=10, gamma=0.5),
            # 'scheduler_params': dict(base_lr=1e-6, max_lr=2e-2, cycle_momentum=False),
            'device_name': 'auto'}

    def optimize_hyperparameters(self, df: pd.DataFrame, feature_col: list, target_col: str, output_file: str) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(TabNetOptunaObjective(df[:5000], feature_col, target_col, self.random_state, self.default_params()), n_trials=20)
        
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
            params['seed'] = self.random_state
            params['n_d'] = model_params['n_d']
            params['n_a'] = model_params['n_d']
            params['gamma'] = model_params['gamma']
            params['momentum'] = model_params['momentum']
            params['n_steps'] = model_params['n_steps']
            params['n_shared'] = model_params['n_shared']
            params['n_independent'] = model_params['n_independent']

            logging.info(f'Parameters used for TabNet supervised training: {params}')

            unsupervised_model = TabNetPretrainer(**params)
            unsupervised_model.fit(X_train=X_train, eval_set=[X_valid], pretraining_ratio=0.5, max_epochs=20)

            model = TabNetRegressor(**params)
            model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_valid, y_valid)],
                eval_name=['valid'],
                eval_metric=['rmse'],
                max_epochs=100,
                patience=10,
                batch_size=1024,
                from_unsupervised=unsupervised_model)

            oof[valid_idx] = model.predict(X_valid).squeeze()
            cv_preds += model.predict(X_test).squeeze() / n_folds
            logging.info(f'Finished fold with score {rmse(y_valid, oof[valid_idx])}')
        
        rmse_score = rmse(df[target_col], oof)
        return rmse_score, cv_preds

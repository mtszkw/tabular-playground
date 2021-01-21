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
    def __init__(self, df, feature_col, target_col, random_state):
        self.df = df
        self.feature_col = feature_col
        self.target_col = target_col
        self.random_state = random_state


    def __call__(self, trial):
        df_train, df_valid = train_test_split(self.df, test_size=0.15, random_state=self.random_state)
        X_train, y_train = df_train[self.feature_col], df_train[self.target_col]
        X_valid, y_valid = df_valid[self.feature_col], df_valid[self.target_col]
        logging.info(f'Train/valid split: {X_train.shape[0]} for training, {X_valid.shape[0]} for validation')
        
        n_d = trial.suggest_int('n_d', 8, 64)
        max_lr = trial.suggest_float('max_lr', 1e-4, 5e-2)

        model = TabNetRegressor(
            n_d=n_d,
            n_a=n_d,
            n_steps=trial.suggest_int('n_steps', 3, 10),
            gamma=trial.suggest_float('gamma', 1.0, 2.0),
            momentum=trial.suggest_float('momentum', 0.01, 0.4),
            optimizer_params=dict(lr=max_lr, weight_decay=1e-5),
            scheudler_params=dict(base_lr=1e-6, max_lr=max_lr, cycle_momentum=False)
        )

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['rmse'],
            max_epochs=500,
            patience=50,
            batch_size=1024)

        score = rmse(y_valid, model.predict(X_valid).squeeze())
        return score


class TabNetTrainer:
    def __init__(self, random_state: int):
        self.random_state = random_state


    def optimize_hyperparameters(self, df: pd.DataFrame, feature_col: list, target_col: str, output_file: str) -> dict:
        study = optuna.create_study(direction='minimize')
        study.optimize(TabNetOptunaObjective(df, feature_col, target_col, self.random_state), n_trials=25)
        
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
            X_train, y_train = df[feature_col].iloc[train_idx], df[target_col].iloc[train_idx]
            X_valid, y_valid = df[feature_col].iloc[valid_idx], df[target_col].iloc[valid_idx]

            model_params['n_estimators'] = 5000
            model_params['learning_rate'] = 1e-2
            
            model = LGBMRegressor(**model_params)
            model.fit(X_train, y_train, eval_set=((X_valid,y_valid)), early_stopping_rounds=500, verbose=0)
            oof[valid_idx] = model.predict(X_valid)
            cv_preds += model.predict(df_test[feature_col]) / n_folds
        
        rmse_score = rmse(df[target_col], oof)
        return rmse_score, cv_preds

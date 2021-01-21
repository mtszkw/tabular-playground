import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMRegressor
import optuna
import optuna.integration.lightgbm as lgb

from metrics import rmse


class LightGBMTrainer:
    def __init__(self, random_state: int):
        self.random_state = random_state


    def optimize_hyperparameters(self, df: pd.DataFrame, feature_col: list, target_col: str, output_file: str) -> dict:
        df_train, df_valid = train_test_split(df, test_size=0.1, random_state=self.random_state)
        X_train, y_train = df_train[feature_col], df_train[target_col]
        X_valid, y_valid = df_valid[feature_col], df_valid[target_col]
        logging.info(f'Train/valid split: {X_train.shape[0]} for training, {X_valid.shape[0]} for validation')

        # Optuna + LightGBM configuration
        train_ds = optuna.integration.lightgbm.Dataset(X_train, label=y_train)
        valid_ds = optuna.integration.lightgbm.Dataset(X_valid, label=y_valid)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "boosting_type": "gbdt",
            "seed": self.random_state
        }

        # HPO
        logging.info('Running optuna.integration.lightgbm.train...')
        optuna_model = optuna.integration.lightgbm.train(
            params,
            train_ds,
            valid_sets=[valid_ds],
            verbose_eval=500,
            early_stopping_rounds=500,
            num_boost_round=20000)
        logging.info(f'Best LightGBM parameters found: {optuna_model.params}')

        with open(output_file, "w") as f:
            f.write(json.dumps(optuna_model.params))

        return optuna_model.params


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

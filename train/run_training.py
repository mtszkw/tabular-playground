import logging
import os

import mlflow
import numpy as np
import pandas as pd

from LightGBM import LightGBMTrainer
from TabNet import TabNetTrainer


config = {
    'folds': 4,
    'seed': 42,
    'lgbm_hpo_run': False,
    'tabnet_hpo_run': False
}


def read_train_test_data(dir: str) -> (pd.DataFrame, pd.DataFrame):
    logging.info('Reading training and test data frames')
    df = pd.read_csv(os.path.join(dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(dir, 'test.csv'))
    logging.info(f'Found {df.shape} rows in training and {df_test.shape} rows in test set')
    return (df, df_test)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    (df, df_test) = read_train_test_data(dir='./input/tabular-playground-series-jan-2021')

    target_col  = 'target'
    feature_col = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
                   'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']

    lgbm_trainer = LightGBMTrainer(random_state=config['seed'])

    if config['lgbm_hpo_run']:
        logging.warning("LightGBM HPO is enabled, starting...")
        lgbm_params = lgbm_trainer.optimize_hyperparameters(df, feature_col, target_col, output_file='lightGBM_params.json')
    else:
        logging.warning(f"LightGBM HPO is disabled, loading params from lightGBM_params.json")
        with open('lightGBM_params.json') as f:
            lgbm_params = json.load(f)

    mlflow.lightgbm.autolog()

    with mlflow.start_run():
        # K-fold cross-validation with optimized parameters
        score, cv_preds = lgbm_trainer.crossval_and_predict(config['folds'], df, df_test, feature_col, target_col, lgbm_params)
        logging.info(f'RMSE on training data = {np.round(score, 7)}')
        logging.info('Finished.')

        mlflow.log_params(lgbm_params)
        mlflow.log_param("folds", config['folds'])
        mlflow.log_param("seed", config['seed'])
        mlflow.log_metrics({"cv_rmse": score})

    df_sub = pd.read_csv('../input/tabular-playground-series-jan-2021/sample_submission.csv')
    df_sub['target'] = cv_preds
    df_sub.to_csv('lightGBM_submission.csv')


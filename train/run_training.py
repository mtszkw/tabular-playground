"""Training.

Usage:
  run_training.py --seed=<seed> --folds=<folds> [--lightgbm] [--tabnet] [--quick] [--skip_hpo]
  run_training.py (-h | --help)
  run_training.py --version

Options:
  -h --help         Show this screen.
  --version         Show version.
  --seed=<seed>     Random state seed [default: 42].
  --folds=<folds>   Number of CV folds [default: 5].
  --lightgbm        LightGBM training.
  --tabnet          TabNet training.
  --quick           Quick run with only short subset of data used.
  --skip_hpo        Whether or not skip HPO, if True: loads lightGBM_params.json or tabnet_params.json.
"""

import json
import logging
import os

from docopt import docopt
import mlflow
import numpy as np
import pandas as pd

import torch
from read_train_test_data import *
from LightGBM import LightGBMTrainer
from TabNet import TabNetTrainer


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    print(torch.cuda.is_available())
    # torch.cuda.set_device(0)
    raise Exception()
    
    arguments = docopt(__doc__, version='Tabular Playground 1.0')
    logging.info(arguments)

    if not any([arguments['--lightgbm'], arguments['--tabnet']]):
        logging.error('Both LightGBM and TabNet are disabled, doing nothing.')

    (df, df_test) = read_train_test_data(dir='./input/tabular-playground-series-jan-2021', quick=arguments['--quick'])
    print(df.head(), df.shape)

    target_col  = 'target'
    feature_col = ['cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7',
                   'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14']

    with mlflow.start_run():
        if arguments['--lightgbm']:
            logging.warning("LightGBM HPO is enabled, starting...")
            lgbm_trainer = LightGBMTrainer(random_state=int(arguments['--seed']))
            lgbm_params = lgbm_trainer.optimize_hyperparameters(df, feature_col, target_col, output_file='lightGBM_params.json')
            mlflow.log_artifact("lightGBM_params.json")

            lgbm_rmse, lgbm_preds = lgbm_trainer.crossval_and_predict(int(arguments['--folds']), df, df_test, feature_col, target_col, lgbm_params)
            logging.info(f'LightGBM RMSE on training data = {np.round(lgbm_rmse, 7)}')

            df_sub = pd.read_csv('./input/tabular-playground-series-jan-2021/sample_submission.csv')
            df_sub['target'] = lgbm_preds
            df_sub.to_csv("lightGBM_submission.csv", index=False)

            mlflow.log_params(lgbm_params)
            mlflow.log_param("folds", arguments['--folds'])
            mlflow.log_param("seed", arguments['--seed'])
            mlflow.log_metrics({"lgbm_cv_rmse": lgbm_rmse})
            mlflow.log_artifact("lightGBM_submission.csv")

        if arguments['--tabnet']:
            tabnet = TabNetTrainer(random_state=int(arguments['--seed']))
            
            if not arguments['--skip_hpo']:
                logging.warning('Option --skip_hpo is set to False, running full hyper-parameter optimization')
                tabnet_params = tabnet.optimize_hyperparameters(df, feature_col, target_col, output_file='tabnet_params.json')
            else:
                logging.warning('Option --skip_hpo is set to True, loading params from tabnet_params.json')
                with open("tabnet_params.json") as f:
                    tabnet_params = json.load(f)

            mlflow.log_artifact("tabnet_params.json")
            tabnet_rmse, tabnet_preds = tabnet.crossval_and_predict(int(arguments['--folds']), df, df_test, feature_col, target_col, tabnet_params)
            logging.info(f'TabNet RMSE on training data = {np.round(tabnet_rmse, 7)}')

            df_sub = pd.read_csv('./input/tabular-playground-series-jan-2021/sample_submission.csv')
            df_sub['target'] = tabnet_preds
            df_sub.to_csv("tabnet_submission.csv", index=False)

            mlflow.log_params(tabnet_params)
            mlflow.log_param("folds", arguments['--folds'])
            mlflow.log_param("seed", arguments['--seed'])
            mlflow.log_metrics({"tabnet_cv_rmse": tabnet_rmse})
            mlflow.log_artifact("tabnet_submission.csv")




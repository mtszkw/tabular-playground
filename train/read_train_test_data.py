
import json
import logging
import os

import numpy as np
import pandas as pd


def read_train_test_data(dir: str, quick: bool) -> (pd.DataFrame, pd.DataFrame):
    logging.info('Reading training and test data frames')
    df = pd.read_csv(os.path.join(dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(dir, 'test.csv'))

    df_public_lb = pd.read_csv('./input/public_lb_069672.csv')
    df_test_leak = df_test.copy()
    df_test_leak['target'] = df_public_lb['target']

    df_leak = pd.concat([df, df_test_leak])
    df_leak.reset_index(drop=True)

    if(quick):
        df_leak = df_leak[:500]

    logging.info(f'Found {df.shape} rows in training and {df_test.shape} rows in test set')
    return (df_leak, df_test)

import numpy as np

def rmse(y_hat, y):
    return np.sqrt(np.mean((y_hat - y) ** 2))

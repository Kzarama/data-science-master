"""Custom metric functions"""

from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import numpy as np


def get_linear_metrics(y_true, y_pred, n=0, k=1):
    """ Calculate the metrics for a linear regression model

    Args:
        y_true (array): true values
        y_pred (array): predicted values
        n (int): number of observations
        k (int): number of features

    Returns:
        r2 (float): r2 score
        mse (float): mean squared error
        rmse (float): root mean squared error
        mae (float): mean absolute error
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    r2_adjusted = 1 - ((1-r2)*(n-1)/(n-k-1))
    mae = np.mean(np.abs(y_true - y_pred))
    return {
        "rmse": rmse,
        "r2": r2,
        "r2_adjusted": r2_adjusted,
        "mae": mae
    }

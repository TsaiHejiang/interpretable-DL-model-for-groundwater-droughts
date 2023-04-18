"""
This file is part of the accompanying code to our paper for reviewing:
"Explaining the mechanism of multiscale groundwater drought events: A new perspective from interpretable deep learning model"

"""

import numpy as np
from scipy import signal
from tensorflow.keras import backend as K

def cal_nse(obs, sim):
    """
    Calculate Nash-Sutcliffe model efficinecy.

    Parameters
    ----------
    obs: observed data.
    sim: simulation data.

    Returns
    ----------
    nse: Nash-Sutcliff model efficiency
    """

    # compute numerator and denominator
    numerator   = np.nansum((obs - sim)**2)
    denominator = np.nansum((obs - np.nanmean(obs))**2)
    # compute coefficient
    return 1 - (numerator / denominator)


def mare(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / np.abs(y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

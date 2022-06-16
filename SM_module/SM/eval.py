#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# Created by : Swamini Khurana
# Created on : On Wed Jun 02 2021 at 21:20:00
# ======================================================================
# __author__ = Swamini Khurana
# __copyright__ = Copyright (c) 2021, Swamini Khurana, Soil Moisture Project
# __credits__ = [Swamini Khurana, Pia Ebeling]
# __license__ = MIT
# __version__ = 0.0.1
# __maintainer__ = Swamini Khurana
# __email__ = swamini.khurana@ufz.de
# __status__ = development
# ======================================================================
""" The file has been build for evaluating performance statistics of models """
#

import numpy as np
import itertools
from scipy import stats
import pandas as pd


def mean_error(y_true=0, y_pred=0):
    """
    Function to return mean of error between true values and predicted values.

    Parameter
    ---------
    y_true: list of true or test values (data type - float).
    y_pred: list of predictions of the model (data type - float).
        
    Returns
    -------
    Mean error (data type - float)
    """

    error = y_pred - y_true
    return np.mean(error)


def residuals(y_true, y_pred):
    """
    Function to return array of residuals for one particular model.

    Parameter
    ---------
    y_true: list of true or test values (data type - float).
    y_pred: list of predictions of the model (data type - float).
        
    Returns
    -------
    2D Numpy array of true values, predicted values and residuals (data type - float)
    """
    error = np.asarray(y_pred) - np.asarray(y_true)
    residuals_array = np.concatenate(
        (
            np.asarray(y_true).reshape(-1, 1),
            np.asarray(y_pred).reshape(-1, 1),
            error.reshape(-1, 1),
        ),
        axis=1,
    )
    return residuals_array


def pair_wise_f_test(data, ycolumn):
    """
    Function to calculate F-statistic as used in APpelhans(2014).
    Returns pair-wise information in F-statistic.
    
    Parameter
    --------
    data: Pandas dataframe with at leasta one column "UID" (string).
    ycolumn: string, column header of values to compare in the dataset containing datatype float.

    Returns
    -------
    DataFrame with columns:
        UID_1: string, UID
        UID_2: string, UID
        F_statistic: float
        p_value: float
    """

    print(
        "Identifying combinations of all models for which you want to calculate f statistic"
    )

    all_models = data.UID.unique().tolist()  # list of all UIDs
    # model_pairs = pair_wise_models(all_models)
    model_pairs = itertools.combinations(all_models, 2)

    print(model_pairs)
    print("Computing F-statistic for: " + ycolumn)
    row = []
    for each_pair in model_pairs:
        model_0 = data[data.UID == each_pair[0]][ycolumn]
        model_1 = data[data.UID == each_pair[1]][ycolumn]
        f_val, p_val = fstatistic(model_0, model_1)
        row.append([each_pair[0], each_pair[1], f_val, p_val])
    results_df = pd.DataFrame.from_records(
        row, columns=["UID_1", "UID_2", "F_statistic", "p_value"]
    )

    return results_df


def fstatistic(x, y):
    """
    Function to calculate F-statistic as used in APpelhans(2014).
    Returns pair-wise information in F-statistic.
    
    Parameter
    --------
    x: Pandas Series with values (float) belonging to model#1
    y: Pandas Series with values (float) belonging to model#2
    
    Returns
    -------
    F statistic and associated p-value
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # f = np.var(x, ddof = 1)/np.var(y,ddof=1) #f value
    f = np.sum(x ** 2) / np.sum(y ** 2)  # f value
    df1 = x.size - 1  # degrees of freedom of model#1
    df2 = y.size - 1  # degrees of freedom of model#2
    p = 1 - stats.f.cdf(f, df1, df2)  # identify p value

    return f, p

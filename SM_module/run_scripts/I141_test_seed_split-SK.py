# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  SWAMINI KHURANA <swamini.khurana@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
import os

# change dir to where the module folder is located
# cur_dir = "C:/Users/khurana/Documents/Scripts/ml-cafe_project_soilmoisture/SM_module"
# os.chdir(cur_dir)

# import itertools
from sklearn.svm import SVR

from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "SK"
method_name = "SVR_100iter"
details = "seed"
# date for which the map should be predicted and plotted
date = "2012-10-22"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "C:/Users/swami/Documents/Projects/soil_moisture"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = SVR(C=0.1, epsilon=0.05, kernel="rbf", max_iter=100)
# define features to use
features = []  # if None, default is used
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with SpatioTempModel, it will load the model named
# accodring to the mandatory variables from saved files.
# create instance of SpatioTempModel

# Train for multiple seeds:
for s in [42, 1337, 7, 420, 12000]:
    method = method_name + "_" + str(s)
    st_model = SpatioTempModel(
        estimator=estimator,
        method=method,
        project_dir=project_dir,
        name=name,
        details=details,
    )
    # train the model
    st_model.train_test(splitseed=s)
    # plot true vs predicted values
    st_model.scatter_plot()
    # plot boxplot of residuals
    st_model.box_plot()
    # record feature importance
    st_model.box_plot_feature_importance()

# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

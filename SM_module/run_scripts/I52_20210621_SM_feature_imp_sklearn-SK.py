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
cur_dir = "C:/Users/khurana/Documents/Scripts/ml-cafe_project_soilmoisture/SM_module"
os.chdir(cur_dir)

from sklearn.svm import SVR
from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "SK"
method = "SVR_fisk"
details = "feature_importance_sklearn"
# date for which the map should be predicted and plotted
date = "2012-21-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "Z:/project_soilmoisture/"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = SVR()
# define features to use
features = ["z", "Clay%", "Sand%", "ele_dem"]  # if None, default is used
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with SpatioTempModel, it will load the model named
# accodring to the mandatory variables from saved files.
# create instance of SpatioTempModel
st_model = SpatioTempModel(
    estimator=estimator,
    method=method,
    name=name,
    project_dir=project_dir,
    details=details,
    features=features,
)
# train the model
st_model.train_test()
# make scatter plot
st_model.scatter_plot()
# show residuals
st_model.residuals
# make box plot of residuals for this model
st_model.box_plot()
# test the feature importance
st_model.imp_features
# Plot the feature importance
# taking features from the project configuration (base list)
st_model.box_plot_feature_importance()
# Plot the feature importance
# taking features from the run script
st_model.box_plot_feature_importance(features)
# Plot the feature importance
# taking features from the run script.
# This should give an error that the dimensions X,Y don't match.
# This is because the feature list provided does not match the training feature list.
st_model.box_plot_feature_importance(["z", "Clay%", "Sand%"])

# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  JULIA SCHMID <julia.schmid@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
import os

# change dir to where the module folder is located
cur_dir = ".."
os.chdir(cur_dir)

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from SM.process import merge_csv_files
from SM.vis import plot_boxplot
from SM.vis import compare_perform_residualmaps
from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "JuliaSchmid"
method = "RF"
details = "boxplots_for_multiple_SpatioTempModels"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "C:/Users/schmidj/Documents/schmidj/DatasciencePhDGroup/ml-cafe_project_soilmoisture/data/"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = RandomForestRegressor()
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

# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

# ------------------------------------------------------------------------------
# RESIDUALS
# ------------------------------------------------------------------------------
# Define directories where you want to search for files to combine/consolidate
# Define directory and filename where you want to output the consolidated data.

residuals_dir = os.path.join(project_dir, "residuals")
output_file = os.path.join(project_dir, "results", "consolidated_residuals.csv")

residuals_data = merge_csv_files(residuals_dir, output_file)

# load residuals data and filter as you see fit
residuals_data = pd.read_csv(output_file)
filt_data = residuals_data

# ------------------------------------------------------------------------------
# SPATIAL MAPS OF RESIDUALS
# ------------------------------------------------------------------------------
#
compare_perform_residualmaps(filt_data, project_dir)
compare_perform_residualmaps(filt_data, project_dir, "r2")
compare_perform_residualmaps(filt_data, project_dir, "rmse")
compare_perform_residualmaps(filt_data, project_dir, "cv")

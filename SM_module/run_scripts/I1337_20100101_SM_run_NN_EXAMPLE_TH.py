# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  FIRSTNAME LASTNAME <EMAILADRESS>
#           FIRSTNAME LASTNAME <EMAILADRESS>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ X."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
from SM.cfg import Project
from SM.training import TrainDailyModel
from sklearn.neural_network import MLPRegressor
from SM.maps import SpatialMap
from SM.misc import clear_results
import os

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "YOURNAME"
method = "YOURMETHOD"
details = "DETAILS"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data. If none
# or if running on EVE this variable will be neglected and standard EVE data
# paths will be used.
# Drop the input data in the project directory in a folder called "model_input"
project_dir = "/Users/houben/phd/ml-cafe/ml_application/SM-module-test-20210129"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
hidden_size = (7, 4)
model = MLPRegressor(
    hidden_layer_sizes=hidden_size,
    learning_rate_init=0.01,
    tol=0.1,
    solver="adam",
    power_t=0.5,
    n_iter_no_change=10,
)
# define features to use
features = ["Date", "Soil_moisture", "Clay%", "Sand%", "ele_dem"]
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with TrainDailyModel, it will load the model named
# accodring to the mandatory variables from saved files.
# create instance of TrainDailyModel
daily_model = TrainDailyModel(
    model=model,
    method=method,
    name=name,
    project_dir=project_dir,
    details=details,
    features=features,
)
# train the model
daily_model.fit()
# make scatter plot
daily_model.scatter_plot()
# make metrics plots
daily_model.metrics_ts_plot()
# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

# ------------------------------------------------------------------------------
# MAP CREATION
# ------------------------------------------------------------------------------
# create instance of SpatialMap with or WITHOUT csv_file
# If csv_file != None it will load the respective file and plot the map based on it
# If csv_file == None it will take the trained model.
# map_3d = SpatialMap(name = name, method = method, date = date, csv_file=csv_file)
map_3d = SpatialMap(name=name, method=method, date=date)
# plot the map
# THIS FUNCTION DOES NOT WORK CURRENTLY: COMPATIBILITY HAS BEEN BROKEN
map_3d.plot_maps(grid=False)

# save the predicted map as csv file
# map_3d.save_csv()

# ------------------------------------------------------------------------------
# CLEAN UP
# ------------------------------------------------------------------------------
# Delete all results.
# clear_results(name = name, method=method)

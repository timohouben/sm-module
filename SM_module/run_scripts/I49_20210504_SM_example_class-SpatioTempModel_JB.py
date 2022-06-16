# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  FIRSTNAME LASTNAME <EMAILADRESS>
#           FIRSTNAME LASTNAME <EMAILADRESS>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ X."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
import os

# change dir to where the module folder is located
os.chdir("..")

from sklearn.ensemble import RandomForestRegressor

from SM.cfg import Project
from SM.maps import SpatialMap
from SM.misc import clear_results
from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "JohannesBoog"
method = "RF"
details = "test_for_SpatioTempModel"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "/home/boog/ufz/12_mlcafe_prj_dsmm/data/test-spatiotemp_sch/"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = RandomForestRegressor()
# define features to override default features
features = ["z", "Clay%", "Sand%", "ele_dem"]
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
# you can set 'train_with_cv=True' to train with 5-fold cross validation
# you can optionally define hyperparameters for tuning
hyperparameters = {"n_estimators": [2, 10], "min_samples_leaf": [2, 4]}
st_model.train_test(train_with_cv=True, tuning_hyperparameters=hyperparameters)
# make scatter plot
st_model.scatter_plot()

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
map_3d = SpatialMap(name=name, method=method, date="2012-09-16")
# plot the map
map_3d.plot_maps(grid=False)

# save the predicted map as csv file
map_3d.save_csv()

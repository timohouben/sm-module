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


from SM.maps import SpatialMap
from SM.misc import clear_results
from SM.training import SpatioTempModel
from SM.cfg import Project

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "PiaEbeling"
method = "RF"
details = "test_for_SpatioTempModel_withFeatureOption"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "C:/Users/ebelingp/Dateien/PhDTeam4DataScience/MLproject/ml-cafe_project_soilmoisture/data/"
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

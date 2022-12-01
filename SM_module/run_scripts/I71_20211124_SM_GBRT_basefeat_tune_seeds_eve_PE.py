# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  PIA EBELING <pia.ebeling@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

This script is used to test the initial train/test split with different seeds. 
The tuning is still done, but in reduced format based on previous optimal tuning
"""
import os

# change dir to where the module folder is located
os.chdir("..")

from sklearn.ensemble import GradientBoostingRegressor


from SM.maps import SpatialMap
from SM.misc import clear_results
from SM.training import SpatioTempModel
from SM.cfg import Project

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "PE"
method_name = "GBRT_seed"
details = "BaseFeatures_tuneShort_seeds"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
#project_dir = "C:/Users/ebelingp/Dateien/PhDTeam4DataScience/MLproject/ml-cafe_project_soilmoisture/data/"
project_dir = "/gpfs1/work/ebelingp/ml-cafe_project_soilmoisture/data/"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = GradientBoostingRegressor()
# define features to use
features = None  # if None, default is used
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with SpatioTempModel, it will load the model named
# according to the mandatory variables from saved files.
# create instance of SpatioTempModel
#Train for multiple seeds:
for s in [42, 1337, 7, 420, 12000]:
    method = method_name + "_" + str(s)
    st_model = SpatioTempModel(
        estimator=estimator,
        method=method,
        name=name,
        project_dir=project_dir,
        details=details,
        features=features,
    )
    # train the model
    #st_model.train_test()
    hyperparameters = {'n_estimators': [200,300,400],'max_depth': [7], "learning_rate": [0.15]} #,"min_samples_split": [3],'loss': ['ls']
    st_model.train_test(splitseed=s, train_with_cv=True, tuning_hyperparameters=hyperparameters)
    # print tuned hyperparameters
    st_model.model.n_estimators
    st_model.model.max_depth
    # make scatter plot of observed versus predicted SM
    st_model.scatter_plot()
    # show residuals
    st_model.residuals
    # make box plot of residuals
    st_model.box_plot()
    #record feature importance
    st_model.box_plot_feature_importance()
    
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

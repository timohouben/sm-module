# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  TIMO HOUBEN <timo.houben@ufz.de>
# ------------------------------------------------------------------------------
import datetime
now = datetime.datetime.now()
print("Starting script at: ")
print(str(now))
import os
import sys

from sklearn.neural_network import MLPRegressor

from SM.maps import SpatialMap
from SM.process import merge_csv_files
from SM.vis import plot_boxplot
from SM.vis import compare_perform_residualmaps
from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# COMMENTS
# taking the best confguration from hyperparameter tuning from model 54
# fitting a model for every seed, 7, 42, 420, 1337, 12000
# ------------------------------------------------------------------------------
# mandatory variables
name = "TimoHouben"
method = str(sys.argv[1])
details = "best_nov22"
# date for which the map should be predicted and plotted
dates = ["2012-21-16", "2012-9-16"]
# copy run script to directory
# ------------------------------------------------------------------------------
# Specify the path to the project directory containing the input data in a
# subfolder called 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "/work/houben/nn-daily"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
mplreg = MLPRegressor(
                    activation="logistic",
                    alpha=1e-05,
                    early_stopping=True,
                    hidden_layer_sizes=(50, 25, 12, 7, 4),
                    learning_rate="adaptive",
                    learning_rate_init=1e-05,
                    max_iter=1000,
                    momentum=0.9,
                    n_iter_no_change=10,
                    nesterovs_momentum=True,
                    random_state=42,
                    shuffle=True,
                    solver="adam",
                    tol=1e-06,
                    validation_fraction=0.1,
                    verbose=True
                    )
# define features to use
features = None  # if None, default is used
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with SpatioTempModel, it will load the model named
# accodring to the mandatory variables from saved files.
# create instance of SpatioTempModel
st_model = SpatioTempModel(
    estimator=mplreg,
    method=method,
    name=name,
    project_dir=project_dir,
    details=details,
    features=features,
)
# train the model
print("Started training...")
# seeds: 7, 42, 1337, 420, 12000
seed = 1337
st_model.train_test(train_with_cv=False, splitseed=seed)
# make scatter plot
st_model.scatter_plot()
# show residuals
st_model.residuals
# make box plot of residuals for this model
st_model.box_plot()
#record feature importance
st_model.box_plot_feature_importance()

# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

print("Training finished...")

# ------------------------------------------------------------------------------
# MAP CREATION
# ------------------------------------------------------------------------------
# create instance of SpatialMap with or WITHOUT csv_file
# If csv_file != None it will load the respective file and plot the map based on it
# If csv_file == None it will take the trained model.
# map_3d = SpatialMap(name = name, method = method, date = date, csv_file=csv_file)

for date in dates:
    map_3d = SpatialMap(name=name, method=method, date=date)
    # plot the map
    map_3d.plot_maps(grid=False)
    # save the predicted map as csv file
    map_3d.save_csv()


# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------
print("Calculating feature importances...")
st_model.box_plot_feature_importance()

then = datetime.datetime.now()
print("Finished script at: ")
print(str(then))
print("Took " + str(then - now) + " to run the script.")

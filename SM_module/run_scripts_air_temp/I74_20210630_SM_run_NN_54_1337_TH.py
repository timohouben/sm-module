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
# using the same configuration as model 51
# searching for best Nn architecture
# ------------------------------------------------------------------------------
# mandatory variables
name = "TimoHouben"
method = '54_1337'
details = ""
# date for which the map should be predicted and plotted
date = "2012-21-16"
# copy run script to directory
# ------------------------------------------------------------------------------
# Specify the path to the project directory containing the input data in a
# subfolder called 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "/work/houben/20240322-ml-cafe/rerun-air_temp"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
hidden_layer_sizes = (9, 7, 4)
# config 1
mplreg = MLPRegressor(
                    activation='relu',
                    alpha=0.0001,
                    batch_size='auto',
                    beta_1=0.9,
                    beta_2=0.999,
                    early_stopping=False,
                    epsilon=1e-08,
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_fun=15000,
                    max_iter=200,
                    momentum=0.9,
                    n_iter_no_change=10,
                    nesterovs_momentum=True,
                    power_t=0.5,
                    random_state=42,
                    shuffle=True,
                    solver='adam',
                    tol=0.0001,
                    validation_fraction=0.1,
                    verbose=True,
                    warm_start=False,
                    )
# define features to use
features = None  # if None, default is used


hidden_layer_sizes = [
    (5, 2),
    (7, 4),
    (9, 7, 4),
    (9, 7, 3),
    (9, 7, 2),
    (12, 5, 2),
    (25, 12, 5, 2),
    (25, 12, 7, 4),
    (50, 25, 12, 7, 4),
    (50, 25, 12, 6, 2),
    (50, 40, 30, 10, 7, 4),
    (50, 40, 30, 10, 4, 2),
    (50, 40, 30, 10, 5, 3),
    (50, 40, 30, 10, 7, 4),
    (25, 50, 50, 10, 7, 4),
    (25, 50, 50, 10, 9, 2),
    (25, 50, 50, 10, 9, 2),
    (15, 20, 20, 10, 9, 2),
    (15, 20, 20, 10, 7, 4),
    (15, 20, 20, 5, 7, 3),
    (15, 20, 20, 15, 7, 3),
    (5, 15, 20, 30, 30, 15, 7, 3),
    (5, 15, 20, 30, 30, 15, 9, 4),
]



# hyperparameter tuning
hyperparameters={
                    #'solver':['lbfgs','sgd', 'adam'],
                    'solver':['adam'],
                    #'hidden_layer_sizes':[(100,),(100, 50, 4), (7, 4), (9, 7, 4)],
                    #'hidden_layer_sizes':[(100, 50, 4), (7, 4), (9, 7, 4)],
                    'hidden_layer_sizes':hidden_layer_sizes,
                    #'activation':['relu','logistic', 'tanh'],
                    'activation':['logistic'],
                    #'activation':['logistic'],
                    #'alpha':[0.0001, 0.001, 0.01, 0.1],
                    'alpha':[0.00001, 0.0001],
                    #'early_stopping':[True,False],
                    'early_stopping':[True],
                    'learning_rate':['adaptive'],
                    'learning_rate_init':[0.00001],
                    'max_iter':[1000],
                    'momentum':[0.9],
                    'n_iter_no_change':[10],
                    'nesterovs_momentum':[True],
                    'random_state':[42],
                    'shuffle':[True],
                    #'tol':[0.0001,0.001, 0.00001],
                    'tol':[0.000001],
                    'validation_fraction':[0.1],
                    'verbose':[True],                       
                    }
# missing features: air temp or soil temp?, rugg_idx
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
# seeds: 1337, 7, 420, 12000
seed = 1337
st_model.train_test(train_with_cv=True, tuning_hyperparameters=hyperparameters, splitseed=seed)
#st_model.train_test(train_with_cv=True, splitseed=12000)
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
# Analysis
# ------------------------------------------------------------------------------
print("Calculating feature importances...")
st_model.box_plot_feature_importance()

then = datetime.datetime.now()
print("Finished script at: ")
print(str(then))
print("Took " + str(then - now) + " to run the script.")

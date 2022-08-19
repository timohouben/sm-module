# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  SWAMINI KHURANA <swamini.khurana@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
# change dir to where the module folder is located
# cur_dir = "C:/Users/khurana/Documents/Scripts/ml-cafe_project_soilmoisture/SM_module"
# os.chdir(cur_dir)

from sklearn.svm import SVR

from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# date for which the map should be predicted and plotted
date = "2012-21-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "/gpfs1/work/khurana/ml_cafe_proj"
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
estimator = SVR(max_iter=1000)
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# Base features as discussed during meetings
name = "SK"
method_name = "1k"
features_base = [
    "P_mm",
    "PET_mm",
    "Elevation[m]",
    "Silt%",
    "Clay%",
    "Sand%",
    "Porosity%",
    "slope",
    "aspect",
    "Temp",
    "X",
    "Y",
    "Z",
]
print("Testing with features: ")
print(features_base)
# mandatory variables
method = method_name + "_2907_1811"
details = "base"  # features
st_model = SpatioTempModel(
    estimator=estimator,
    method=method,
    name=name,
    details=details,
    project_dir=project_dir,
)
# train the model
st_model.train_test(
    train_with_cv=True,
    tuning_hyperparameters={
        "kernel": ["linear", "rbf"],
        "epsilon": [0.01, 0.05, 0.1, 0.5, 10, 100],
        "C": [0.1, 0.5, 1.0, 1.5, 10, 100],
    },
)
# plot true vs predicted values
st_model.scatter_plot()
# plot boxplot of residuals
st_model.box_plot()
# record feature importance
st_model.box_plot_feature_importance()

# All features
name = "SK"
method_name = "1k"
features = [
    "P_mm",
    "PET_mm",
    "Elevation[m]",
    "Silt%",
    "Clay%",
    "Sand%",
    "Porosity%",
    "dist_to_creek",
    "twi",
    "relief_1",
    "relief_2",
    "relief_3",
    "hillshade",
    "rugg_idx",
    "slope",
    "aspect",
    "ele_dem",
    "Temp",
]
print("Testing with features: ")
print(features_base)
# mandatory variables
method = method_name + "_all_1811"
details = features
st_model1 = SpatioTempModel(
    estimator=estimator,
    method=method,
    name=name,
    details=details,
    features=features,
    project_dir=project_dir,
)
# train the model
st_model1.train_test(
    train_with_cv=True,
    tuning_hyperparameters={
        "kernel": ["linear", "rbf"],
        "epsilon": [0.01, 0.05, 0.1, 0.5, 10, 100],
        "C": [0.1, 0.5, 1.0, 1.5, 10, 100],
    },
)
# plot true vs predicted values
st_model1.scatter_plot()
# plot boxplot of residuals
st_model1.box_plot()
# record feature importance
st_model1.box_plot_feature_importance(features)

# All features
name = "SK"
method_name = "1k_1811"
features_subset = [
    "P_mm",
    "PET_mm",
    "Elevation[m]",
    "Silt%",
    "Clay%",
    "Sand%",
    "Porosity%",
    "dist_to_creek",
    "twi",
    "relief_1",
    "relief_2",
    "relief_3",
    "hillshade",
    "rugg_idx",
    "slope",
    "aspect",
    "ele_dem",
    "Temp",
]
print("Testing with features: ")
print(features_base)
# mandatory variables
method = method_name + "_subset"
details = features
st_model2 = SpatioTempModel(
    estimator=estimator,
    method=method,
    name=name,
    details=details,
    features=features_subset,
    project_dir=project_dir,
)
# train the model
st_model2.train_test(
    train_with_cv=True,
    tuning_hyperparameters={
        "kernel": ["linear", "rbf"],
        "epsilon": [0.01, 0.05, 0.1, 0.5, 10, 100],
        "C": [0.1, 0.5, 1.0, 1.5, 10, 100],
    },
)
# plot true vs predicted values
st_model2.scatter_plot()
# plot boxplot of residuals
st_model2.box_plot()
# record feature importance
st_model2.box_plot_feature_importance(features_subset)

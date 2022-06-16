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

import pandas as pd

import itertools
from sklearn.ensemble import RandomForestRegressor

from SM.process import merge_csv_files
from SM.vis import f_heatmap,temporal_scatter, plot_boxplot, compare_perform_residualmaps, compare_perform_boxplots
from SM.eval import pair_wise_f_test
from SM.training import SpatioTempModel

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "SK"
method = "RF"
details = "boxplots_for_multiple_SpatioTempModels"
# date for which the map should be predicted and plotted
date = "2012-09-16"
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "Z:/project_soilmoisture/"
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
name = "SK"
method_name = "RF"
tot_n = len(features)
#Train for multiple iterations of features:
count = 1
for f_num in list(range(2,tot_n)):
    feature_comb = itertools.combinations(features, f_num)
    for sub_f_list in feature_comb:
        print("Testing with features: ")
        print(list(sub_f_list))
        # mandatory variables
        method = method_name+"_"+str(count)
        details = sub_f_list
        st_model = SpatioTempModel(
            estimator=estimator,
            method=method,
            project_dir = project_dir,
            name=name,
            details=details,
            features=list(sub_f_list),
            )
        # train the model
        st_model.train_test()
        # plot true vs predicted values
        st_model.scatter_plot()
        # plot boxplot of residuals
        st_model.box_plot()
        #record feature importance
        st_model.box_plot_feature_importance(list(sub_f_list))
        count +=1
    
#Select for the models with the best performance statistic
performance_dir = os.path.join(project_dir, "performance_stats")
output_file = os.path.join(project_dir, "results", "consolidated_performance_stats.csv")

performance_data = merge_csv_files(performance_dir, output_file)

#selecting for low rmse:
top_6_models = performance_data.sort_values(by=['R2_score'])[:6]

compare_perform_boxplots(list(top_6_models.UID.values), ['R2_score','RMSE', 'MAE'], project_dir, "SK")

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

# ------------------------------------------------------------------------------
# BOX PLOTS
# ------------------------------------------------------------------------------
#
# load residuals data and filter as you see fit
residuals_data = pd.read_csv(output_file)
filt_data = residuals_data
fig = plot_boxplot(filt_data, "residuals", "UID")
figfile = os.path.join(project_dir, "figures", "residuals_boxplots.png")
fig.savefig(figfile, dpi=300, layout="tight")

# ------------------------------------------------------------------------------
# TIME SERIES SCATTER PLOT OF RESIDUALS
# ------------------------------------------------------------------------------
# Load consolidated file having true values, predictions, residuals
# Compute statistics (variance, error, F statistic) based on this

residuals_dir = os.path.join(project_dir, "results")
filename = "consolidated_residuals.csv"
residuals_path = os.path.join(residuals_dir, filename)
data = pd.read_csv(residuals_path)
filt_data = data
temp_scat = temporal_scatter(filt_data, "residuals", "Date", "y_true", project_dir)

# PERFORMANCE STATISTICS
# ------------------------------------------------------------------------------
# Load consolidated file having true values, predictions, residuals
# Compute statistics (variance, error, F statistic) based on this

residuals_dir = os.path.join(project_dir, "results")
filename = "consolidated_residuals.csv"
residuals_path = os.path.join(residuals_dir, filename)
data = pd.read_csv(residuals_path)
print(data.columns)
filt_data = data
f_df = pair_wise_f_test(filt_data, "residuals")
f_plot = f_heatmap(f_df, "F_statistic")

# ------------------------------------------------------------------------------
# SPATIAL MAPS OF RESIDUALS
# ------------------------------------------------------------------------------
#
compare_perform_residualmaps(filt_data, project_dir)
compare_perform_residualmaps(filt_data, project_dir, "r2")
compare_perform_residualmaps(filt_data, project_dir, "rmse")
compare_perform_residualmaps(filt_data, project_dir, "cv")

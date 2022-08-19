# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  SWAMINI KHURANA <swamini.khurana@gmail.com>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
#%%
import os

# change dir to where the module folder is located
cur_dir = "C:/Users/swami/Documents/Projects/soil_moisture/ml_cafe_proj"
os.chdir(cur_dir)

import pandas as pd

import itertools
from sklearn.svm import SVR

from SM.process import merge_csv_files
from SM.vis import (
    f_heatmap,
    temporal_scatter,
    plot_boxplot,
    compare_perform_residualmaps,
    compare_perform_boxplots,
)
from SM.eval import pair_wise_f_test
from SM.training import SpatioTempModel
from SM.maps import SpatialMap
from SM.io import load_model

print("All libraries loaded")
#%%
# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "SK"
project_dir = "C:/Users/swami/Documents/Projects/soil_moisture/ml_cafe_proj"

# Select for the models with the best performance statistic
performance_dir = os.path.join(project_dir, "performance_stats")
output_file = os.path.join(project_dir, "results", "consolidated_performance_stats.csv")

performance_data = merge_csv_files(performance_dir, output_file)
print("Performance statistics consolidated and saved in csv file")
print(performance_data.shape)
#%%
# Filter data:
performance_data = pd.read_csv(output_file)
top_6_models = performance_data.sort_values(by="R2_score", ascending=False)[:6]
print(top_6_models.shape)
print("Models sorted in descending order by r2 score")

compare_perform_boxplots(
    list(top_6_models.UID.values), ["R2_score", "RMSE", "MAE"], project_dir, "SK"
)
print("Box plots comparing performances saved")
top_6_models.to_csv(
    os.path.join(project_dir, "results", "top_6_models.csv"), index=False
)
toplist = list(top_6_models.UID.values)
# don't change these lines
try:
    csv_file = os.path.abspath(csv_file_path)
except NameError:
    pass

#%%
# ------------------------------------------------------------------------------
# RESIDUALS
# ------------------------------------------------------------------------------
# Define directories where you want to search for files to combine/consolidate
# Define directory and filename where you want to output the consolidated data.

residuals_dir = os.path.join(project_dir, "residuals")
output_file = os.path.join(project_dir, "results", "consolidated_residuals.csv")

#%%
residuals_data = merge_csv_files(residuals_dir, output_file)

#%%
# ------------------------------------------------------------------------------
# BOX PLOTS
# ------------------------------------------------------------------------------
#
# load residuals data and filter as you see fit
residuals_data = pd.read_csv(output_file)
filt_data = residuals_data[residuals_data["UID"].isin(toplist)].reset_index()
fig = plot_boxplot(filt_data, "residuals", "UID")
figfile = os.path.join(project_dir, "figures", "residuals_boxplots.png")
fig.savefig(figfile, dpi=300, layout="tight")

#%%
# ------------------------------------------------------------------------------
# TIME SERIES SCATTER PLOT OF RESIDUALS
# ------------------------------------------------------------------------------
# Load consolidated file having true values, predictions, residuals
# Compute statistics (variance, error, F statistic) based on this

residuals_dir = os.path.join(project_dir, "results")
filename = "consolidated_residuals.csv"
residuals_path = os.path.join(residuals_dir, filename)
data = pd.read_csv(residuals_path)
# filt_data = data
temp_scat = temporal_scatter(filt_data, "residuals", "Date", "y_true", project_dir)

#%%
# PERFORMANCE STATISTICS
# ------------------------------------------------------------------------------
# Load consolidated file having true values, predictions, residuals
# Compute statistics (variance, error, F statistic) based on this

residuals_dir = os.path.join(project_dir, "results")
filename = "consolidated_residuals.csv"
residuals_path = os.path.join(residuals_dir, filename)
data = pd.read_csv(residuals_path)
print(data.columns)
f_df = pair_wise_f_test(filt_data, "residuals")
f_plot = f_heatmap(f_df, "F_statistic")

#%%
# ------------------------------------------------------------------------------
# SPATIAL MAPS OF RESIDUALS
# ------------------------------------------------------------------------------
#
# date for which the map should be predicted and plotted
date = "2012-09-16"
compare_perform_residualmaps(filt_data, project_dir)
compare_perform_residualmaps(filt_data, project_dir, "r2")
compare_perform_residualmaps(filt_data, project_dir, "rmse")
compare_perform_residualmaps(filt_data, project_dir, "cv")

#%%
# ------------------------------------------------------------------------------
# SPATIAL MAPS OF PREDICTIONS
# ------------------------------------------------------------------------------
#
date = "2012-09-16"
for method in toplist:
    print("Predicting map for method", method)
    details = "maps"
    # predict the map
    map_3d = SpatialMap(name=name, method=method[3:], date=date)
    # plot the map
    map_3d.plot_maps(grid=False)
    # save the predicted map as csv file
    map_3d.save_csv()

#%%
# ------------------------------------------------------------------------------
# SPATIAL MAPS OF PREDICTIONS FOR BEST SEED, BUT DIFFERENT DATES
# ------------------------------------------------------------------------------
#
name = "SK"
method = "iter_5000_seed_12000"
dates = ["2012-10-16", "2013-01-16", "2013-04-16", "2013-07-16"]
for date_op in dates:
    # predict the map
    map_3d = SpatialMap(name=name, method=method, date=date_op)
    # plot the map
    map_3d.plot_maps(grid=False)
    # save the predicted map as csv file
    map_3d.save_csv()

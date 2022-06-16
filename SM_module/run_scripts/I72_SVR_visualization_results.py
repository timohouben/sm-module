# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:48:24 2021

@author: khurana
"""
import os
import pandas as pd

# change dir to where the module folder is located
cur_dir = "C:/Users/khurana/Documents/Scripts/ml-cafe_project_soilmoisture/SM_module"
os.chdir(cur_dir)

from SM.process import merge_csv_files
from SM.vis import f_heatmap,temporal_scatter, plot_boxplot, compare_perform_boxplots, compare_perform_residualmaps
from SM.eval import pair_wise_f_test

# date for which the map should be predicted and plotted
date = "2012-21-16"
project_dir = 'Z:/project_soilmoisture'

#Select for the models with the best performance statistic
performance_dir = os.path.join(project_dir, "performance_stats")
output_file = os.path.join(project_dir, "results", "consolidated_performance_stats.csv")

performance_data = merge_csv_files(performance_dir, output_file)

#Filter data:
performance_data = pd.read_csv(output_file)
#select_models = performance_data.sort_values(by=['R2_score'])[:6]
select_models = performance_data[performance_data['Method'].str.contains("base", na = False)]

compare_perform_boxplots(list(select_models.UID.values), ['R2_score','RMSE', 'MAE'], project_dir, "SK")

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
filt_data = residuals_data[residuals_data['UID'].str.contains("base", na=False)]
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
filt_data = data[data['UID'].str.contains("base", na=False)]
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
filt_data = data[data['UID'].str.contains("base", na=False)]
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

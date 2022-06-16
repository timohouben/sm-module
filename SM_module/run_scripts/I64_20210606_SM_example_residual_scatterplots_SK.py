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
from SM.vis import temporal_scatter


# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
# Specify the path to the project directory containing the input data in a
# subfolder calles 'model_input'.
# If none or if running on EVE this variable will be neglected and standard EVE
# data paths will be used
project_dir = "Z:/project_soilmoisture"

# ------------------------------------------------------------------------------
# TIME SERIES SCATTER PLOT OF RESIDUALS
# ------------------------------------------------------------------------------
# Load consolidated file having true values, predictions, residuals
# Compute statistics (variance, error, F statistic) based on this

residuals_dir = os.path.join(project_dir, "results")
filename = "consolidated_residuals.csv"
residuals_path = os.path.join(residuals_dir, filename)
data = pd.read_csv(residuals_path)

temp_scat = temporal_scatter(data, "residuals", "Date", "y_true", project_dir)

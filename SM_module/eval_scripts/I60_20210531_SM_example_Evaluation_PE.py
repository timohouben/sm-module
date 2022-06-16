# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  FIRSTNAME LASTNAME <EMAILADRESS>
#           FIRSTNAME LASTNAME <EMAILADRESS>
# ------------------------------------------------------------------------------
"""Evaluation script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""

from SM.cfg import Project
from SM.vis import compare_perform_boxplots

# ------------------------------------------------------------------------------
# SETTINGS FOR COMPARISON
# ------------------------------------------------------------------------------
# mandatory variables
project_dir = "C:/Users/ebelingp/Dateien/PhDTeam4DataScience/MLproject/ml-cafe_project_soilmoisture/data/"
# model result files to compare
filenames = ["PiaEbeling_RF.csv", "PiaEbeling_RF_dummy.csv", "PiaEbeling_RF_dummy2.csv"]
# measures to compare
measures = ["R2_score", "MAE", "RMSE"]

# ------------------------------------------------------------------------------
# COMPARISON OF MODELS
# ------------------------------------------------------------------------------
compare_perform_boxplots(
    filenames=filenames, measures=measures, project_dir=project_dir, pltname="test"
)

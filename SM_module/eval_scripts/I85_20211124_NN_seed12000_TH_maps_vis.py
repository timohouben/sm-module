# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  PIA EBELING <pia.ebeling@ufz.de>
# ------------------------------------------------------------------------------
import os
from sklearn.neural_network import MLPRegressor
import pandas as pd

# own modules
from SM.maps import SpatialMap
from SM.training import SpatioTempModel
from SM.io import load_model

# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------
# mandatory variables
name = "TimoHouben"
method = "54_12000"
details = ""
# Specify the path to the project directory containing the model to load
# subfolder called 'models' and where results will be saved
project_dir = "/work/houben/20240322-ml-cafe/rerun-air_temp"
# specify the model and its parameters
estimator = MLPRegressor()
# define features to use
features = None  # if None, default is used

# ------------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------------
fname = os.path.join(project_dir, "models", name, method, name + "_" + method)
st_model = SpatioTempModel(
    estimator=estimator,
    method=method,
    name=name,
    project_dir=project_dir,
    details=details,
    features=features,
)

# load the model and save it as attribute of the
st_model.model, _ = load_model(fname)

# ------------------------------------------------------------------------------
# MAP CREATION
# ------------------------------------------------------------------------------
# create maps for several dates
# date for which the maps should be predicted and plotted
# dates = [
#     "2012-09-16",
#     "2012-10-16",
#     "2012-11-16",
#     "2012-12-16",
#     "2013-01-16",
#     "2013-02-16",
#     "2013-03-16",
#     "2013-04-16",
#     "2013-05-16",
#     "2013-06-16",
#     "2013-07-16",
#     "2013-08-16",
#     "2013-09-16",
#     "2013-10-16",
# ]

date_from = '2012-09-16'
date_to = '2013-11-14'
dates = pd.date_range(start=date_from, end=date_to, freq='D')
# convert to string without time
dates = dates.strftime('%Y-%m-%d')

for date in dates:
    try:
        print("Predicting map for method", method, "date", date)
        # predict the map
        map_3d = SpatialMap(name=name, method=method, date=date)
        # plot the map
        map_3d.plot_maps(grid=False)
        # save the predicted map as csv file
        map_3d.save_csv()
    except:
        print("Error for date", date)
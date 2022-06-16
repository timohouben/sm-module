# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  SWAMINI KHURANA <swamini.khurana@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 2."

Leave one blank line and explain the model run in detailed way. Put as much
information as you can/is worth the effort.
"""
# change dir to where the module folder is located
#cur_dir = "C:/Users/khurana/Documents/Scripts/ml-cafe_project_soilmoisture/SM_module"
#os.chdir(cur_dir)

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
project_dir = '/gpfs1/work/khurana/ml_cafe_proj'
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
#Base features as discussed during meetings
name = "SK"
method_name = "SVR_base"
features_base = ["P_mm","PET_mm","Elevation[m]","Silt%","Clay%","Sand%","Porosity%","slope","aspect","Temp","X","Y","Z"]
print("Testing with features: ")
print(features_base)
# mandatory variables
for max_num,max_name in zip([100,1000,5000,10000,50000,100000], [".1k", "1k","5k", "10k","50k","100k"]):
    method = method_name + "_" + max_name
    details = max_name
    estimator = SVR(max_iter=max_num)
    st_model = SpatioTempModel(
                estimator=estimator,
                method=method,
                name=name,
                details=details,
                project_dir = project_dir)
    # train the model
    st_model.train_test(train_with_cv=True, tuning_hyperparameters={'kernel':['linear','rbf'],
                                                                    'epsilon':[0.01,0.05,0.1,0.5,10,100],
                                                                    'C':[0.1,0.5,1.0,1.5,10,100]})
    # plot true vs predicted values
    st_model.scatter_plot()
    # plot boxplot of residuals
    st_model.box_plot()
    #record feature importance
    st_model.box_plot_feature_importance()
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
project_dir = '/work/houben/20240322-ml-cafe/rerun-air_temp'
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
# If no model is trained with SpatioTempModel, it will load the model named
# accodring to the mandatory variables from saved files.
# create instance of SpatioTempModel
#Base features as discussed during meetings
name = "SK"
details = "seed"

# hyperparameters for training
tuning_hyperparameters = {
    "kernel": ["linear", "rbf"],
    "epsilon": [0.01, 0.1, 0.5, 1.0],
    "C": [0.1, 1.0, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1, 10]  # 'scale' and 'auto' are included for RBF kernel
}

#Train for multiple seeds:
#for i in [1000,5000,10000,25000]:
for i in [5000]:
    estimator = SVR(C=0.1, epsilon = 0.05, kernel = 'rbf', max_iter = i)
    method_name= "iter_"+str(i)
    for s in [42, 1337, 7, 420, 12000]:
        method = method_name+"_seed_"+str(s)
        st_model = SpatioTempModel(
	        estimator=estimator,
        	method=method,
	        project_dir = project_dir,
	        name=name,
        	details=details
        	)
        # train the model
        st_model.train_test(train_with_cv=True,
                            tuning_hyperparameters=tuning_hyperparameters,
                            splitseed=s)
        # plot true vs predicted values
        st_model.scatter_plot()
	    # plot boxplot of residuals
        st_model.box_plot()
	    #record feature importance
        st_model.box_plot_feature_importance()
	    # plot true vs predicted values
        st_model.scatter_plot()
	    # plot boxplot of residuals
        st_model.box_plot()
	    #record feature importance
        st_model.box_plot_feature_importance()
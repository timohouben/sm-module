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
#project_dir = 'C:/Users/swami/Documents/Project_Data/soil_moisture' #directory in SK's system
# optional variables: if you want to read map predictions from file and use this
# for plotting
# csv_file_path = "FULL/PATH/TO/A/CSV/FILE/FOR/MAP/CREATION.csv"
# specify the model and its parameters
# ------------------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------------------
#Base features as discussed during meetings
name = "SK"
method_name = "rbf_deg"
features_base = ["P_mm","PET_mm","Elevation[m]","Silt%","Clay%","Sand%","Porosity%","slope","aspect","Temp","X","Y","Z"]
print("Testing with features: ")
print(features_base)
# mandatory variables
estimator = SVR(epsilon = 0.01, C = 0.01, gamma = 1, kernel = 'rbf', degree = 3)
for max_num,max_name in zip([10000,250000,50000,100000], ["10k","25k","50k","100k"]):
	method = method_name + max_name
	details = max_name
	st_model = SpatioTempModel(estimator=estimator,method=method,name=name,details=details,project_dir = project_dir)
	print ("Model loaded: "+ str(method))
	# train the model
	st_model.train_test()#train_with_cv=True, tuning_hyperparameters={'degree':[2,3]})
	print ("Parameters tuned: "+ str(method))
	# plot true vs predicted values
	#st_model.scatter_plot()
	print ("Scatter plot: "+ str(method))
	# plot boxplot of residuals
	#st_model.box_plot()
	print ("Boxplot: "+ str(method))
	#record feature importance
	#st_model.box_plot_feature_importance()
	print ("feature importance: "+ str(method))

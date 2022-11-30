# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Authors:  JOHANNES BOOG <johannes.boog@ufz.de>
# ------------------------------------------------------------------------------
"""Run script for SM module to answer RQ 1."

This model is based on th Random Forest algorithm and estimates soil moisture 
across time and space through the entire data set.
"""
import os
# change dir to where the module folder is located 
os.chdir("..")

from sklearn.ensemble import RandomForestRegressor

from SM.cfg import Project
from SM.maps import SpatialMap
from SM.misc import clear_results
from SM.training import SpatioTempModel
# ------------------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------------------

# mandatory variables
cfg_d = {"name" : 'JB',
         "method" : 'RF_seed_',
         "details" : 'fitting_of_spatiotempmodel_with_rf_seeds',
         # date for which the map should be predicted and plotted
         "dates" : ['2012-09-16', '2013-07-16', '2013-04-16', '2013-01-16'],
         "project_dir" : "/home/boog/ufz/12_mlcafe_prj_dsmm/data/I69-fit_rf",
         "estimator" : RandomForestRegressor(random_state=42, n_estimators=2)}

# define features to override default features
features_d = {"default": None}#,
              #"set1" : ["z", "dayofyear_cos",
              #             "Sand%","Porosity%", "aspect_cos","ele_dem",
              #             "P_mm", "Temp"],
              #"set2" : ["z", "dayofyear_cos",
              #            "Clay%","Sand%","Porosity%",
              #            "slope","ele_dem", "twi",
              #            "P_mm","PET_mm", "Temp"],
              #"set3" : ["z", "dayofyear_cos",
              #             "Sand%","Porosity%", "aspect_cos","ele_dem",
              #             "P_mm"],
              #"set4" : ["z", "P_mm","PET_mm",
              #          "dayofyear_cos", "dayofyear_sin",
              #          "Silt%","Clay%","Sand%","Porosity%",
              #          "dist_to_creek","twi","relief_1","relief_2","relief_3",
              #          "hillshade","rugg_idx","slope","aspect_cos","aspect_sin",
              #          "ele_dem","Temp"]}

seeds_l = [7, 42, 420, 1337, 12000]

# hyperparameters for training
hyperparameters = {'n_estimators': [100, 1000],# 5000], 
                   "min_samples_leaf": [4, 8],#16],
                   "max_features" : [0.33],
                   "warm_start" : [False, True],
                   "n_jobs":[4]}

# TRAINING FUNCTION ------------------------------------------------------------
def run_ensemble(cfg_d, features_d, seeds_l, hyperparameters):
    
    for featset, features in features_d.items():
        for seed in seeds_l:
            
            name=cfg_d["name"]+"_"+str(seed)
            method=cfg_d["method"]
            project_dir=cfg_d["project_dir"]
            details=cfg_d["details"]
            dates=cfg_d["dates"]
            estimator=cfg_d["estimator"]

            st_model = SpatioTempModel(
                        estimator=estimator, method=method, name=name, 
                        project_dir=project_dir, details=details,
                        features=features)
            # train the model        
            st_model.train_test(train_with_cv=True,
                                tuning_hyperparameters=hyperparameters,
                                splitseed=seed)
            # make scatter plot
            st_model.scatter_plot()

            # make box plot of residuals for this model
            st_model.box_plot()

            #record feature importance
            st_model.box_plot_feature_importance()

            for date in dates:            
                map_3d = SpatialMap(name = name, method = method, date=date)
                map_3d.plot_maps(grid = False)
                map_3d.save_csv()

            del st_model, map_3d

        print("Model fitting finished, :-).")
        

# ------------------------------------------------------------------------------
# Let's go
# ------------------------------------------------------------------------------
run_ensemble(cfg_d, features_d, seeds_l, hyperparameters)

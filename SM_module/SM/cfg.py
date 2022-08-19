# -*- coding: utf-8 -*-
""" The file has been build for providing with configurations about the data"""
#
import os

from SM.misc import platform_release


class Project(object):
    """Class to define general information on the modeling project, such as
    features to be used, file paths."""

    # Default features, we can add or remove features here (and below)
    # TODO(JB): - include rugg_idx? (TBC)
    features_default = [
        "Date_integer",
        "z",
        "dayofyear_sin",
        "dayofyear_cos",
        "Silt%",
        "Clay%",
        "Sand%",
        "Porosity%",
        "slope",
        "aspect_sin",
        "aspect_cos",
        "ele_dem",
        "twi",
        "P_mm",
        "PET_mm",
        "Temp",
        "x",
        "y",
    ]

    features_select = features_default

    # coordinate reference system for spatial ops and geoPandas
    crs = "EPSG:32632"

    @classmethod
    def set_features(self, features=None):
        """Method to override the default features.

        Parameter
        ---------
        features : list
            Features to use for this project, must be column names in the data frame.
            The selected features replace the default features.
        """
        if features is None:
            # Default features
            self.features_select = self.features_default
            print("You are using the default features")
        else:
            # individual features
            self.features_select = features
            print("You have set the features")

    @classmethod
    def set_project_paths(self, project_dir=None, project_type="SpatioTempModel"):
        """Define project specific path to input and output data.

        Parameter
        ---------
        project_dir : str, Default None.
            Directory of local project. If None, a default directory will be
            used.
        project_type : str
            Type of model, either "SpatioTempModel" for a spatio-temporal model
            on all data; or "TrainDailyModel" for creating spatial models for
            daily data subsets.

        """

        self.platform = platform_release()
        self.project_type = project_type

        # set default input data
        if self.project_type is "TrainDailyModel":

            input_data = "daily_data_boxes_v1_0.pkl"
            input_raster = "static_data_raster_v1_0.pkl"

        else:  # if self.project_type is "spatiSpatioTempModelotemp":

            input_data = "SCH_smmeteotxtdemtemp_20200919_Master.csv"
            input_raster = "SCH_txtdem_20210412_static_raster_Master.csv"

        # set path
        if (
            project_dir is None and self.platform is "eve"
        ):  # default option for eve-cluster

            self.project_dir = os.path.abspath(
                "/data/ml-cafe/project_soilmoisture/data/schaefertal"
            )
            self.data_path = os.path.join(self.project_dir, "model_input", input_data)
            self.raster_path = os.path.join(
                self.project_dir, "model_input", input_raster
            )
            self.models_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "models"
            )
            self.figures_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "figures"
            )
            self.results_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "results"
            )
            self.residuals_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "residuals"
            )
            self.performance_stats_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module",
                "performance_stats",
            )
            self.hyperparameters_tuning_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module",
                "hyperparameters_tuning_stats",
            )

        else:

            self.project_dir = project_dir
            self.data_path = os.path.join(self.project_dir, "model_input", input_data)
            self.raster_path = os.path.join(
                self.project_dir, "model_input", input_raster
            )
            self.models_path = os.path.join(self.project_dir, "models")
            self.figures_path = os.path.join(self.project_dir, "figures")
            self.results_path = os.path.join(self.project_dir, "results")
            self.residuals_path = os.path.join(self.project_dir, "residuals")
            self.performance_stats_path = os.path.join(
                self.project_dir, "performance_stats"
            )
            self.hyperparameters_tuning_path = os.path.join(
                self.project_dir, "hyperparameters_tuning_stats"
            )

        print(
            "You have changed the project directory from standard (EVE) to {}".format(
                project_dir
            )
        )

    @classmethod
    def set_inputdata(self, in_data_name, raster_name):
        """Temporary solution: overwrite path to input data and input raster.

        in_data_name : str
            Name of the input data file.
        raster_name : str
            Name of the input rasterfile.
        """
        if self.platform is "eve":
            self.data_path = os.path.join(self.project_dir, in_data_name)
            self.raster_path = os.path.join(self.project_dir, raster_name)
        else:
            self.data_path = os.path.join(self.project_dir, "model_input", in_data_name)
            self.raster_path = os.path.join(
                self.project_dir, "model_input", raster_name
            )

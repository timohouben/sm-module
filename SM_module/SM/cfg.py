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
    input_data = "default"
    input_raster = "default"


    # coordinate reference system for spatial ops and geoPandas
    crs = "EPSG:32632"

    @classmethod
    def set_features(cls, features=None):
        """Method to override the default features.

        Parameter
        ---------
        features : list
            Features to use for this project, must be column names in the data frame.
            The selected features replace the default features.
        """
        if features is None:
            # Default features
            cls.features_select = cls.features_default
            print("You are using the default features for SpatioTempmodel.")
        else:
            # individual features
            cls.features_select = features
            print("You have set the features")

    @classmethod
    def set_project_paths(cls, project_dir=None, project_type="SpatioTempModel"):
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
        cls.project_dir = project_dir

        cls.platform = platform_release()
        cls.project_type = project_type

        # set default input data
        if cls.project_type is "TrainDailyModel":
            if cls.input_data == "default":
                cls.input_data = "daily_data_boxes_v1_0.pkl"
            if cls.input_raster == "default":
                cls.input_raster = "static_data_raster_v1_0.pkl"
            print(cls.input_data)
        else:  # if cls.project_type is "spatiSpatioTempModelotemp":

            if cls.input_data == "default":
                cls.input_data = "SCH_smmeteotxtdemtemp_20200919_Master.csv"
            if cls.input_raster == "default":
                cls.input_raster = "SCH_txtdem_20210412_static_raster_Master.csv"



        # set path
        if (
            project_dir is None and cls.platform is "eve"
        ):  # default option for eve-cluster
            cls.project_dir = os.path.abspath(
                "/data/ml-cafe/project_soilmoisture/data/schaefertal"
            )

            cls.data_path = os.path.join(cls.project_dir, "model_input", cls.input_data)
            cls.raster_path = os.path.join(
                cls.project_dir, "model_input", cls.input_raster
            )
            cls.models_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "models"
            )
            cls.figures_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "figures"
            )
            cls.results_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "results"
            )
            cls.residuals_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module", "residuals"
            )
            cls.performance_stats_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module",
                "performance_stats",
            )
            cls.hyperparameters_tuning_path = os.path.join(
                "/data/ml-cafe/project_soilmoisture/results/sm-module",
                "hyperparameters_tuning_stats",
            )

        else:

            cls.data_path = os.path.join(cls.project_dir, "model_input", cls.input_data)
            cls.raster_path = os.path.join(
                cls.project_dir, "model_input", cls.input_raster
            )
            cls.models_path = os.path.join(cls.project_dir, "models")
            cls.figures_path = os.path.join(cls.project_dir, "figures")
            cls.results_path = os.path.join(cls.project_dir, "results")
            cls.residuals_path = os.path.join(cls.project_dir, "residuals")
            cls.performance_stats_path = os.path.join(
                cls.project_dir, "performance_stats"
            )
            cls.hyperparameters_tuning_path = os.path.join(
                cls.project_dir, "hyperparameters_tuning_stats"
            )

        print(
            "You have changed the project directory from standard (EVE) to {}".format(
                project_dir
            )
        )

    # @classmethod
    # def set_inputdata(cls, in_data_name, raster_name):
    #     """Temporary solution: overwrite path to input data and input raster.

    #     in_data_name : str
    #         Name of the input data file.
    #     raster_name : str
    #         Name of the input rasterfile.
    #     """
    #     if cls.platform is "eve":
    #         cls.data_path = os.path.join(cls.project_dir, in_data_name)
    #         cls.raster_path = os.path.join(cls.project_dir, raster_name)
    #     else:
    #         cls.data_path = os.path.join(cls.project_dir, "model_input", in_data_name)
    #         cls.raster_path = os.path.join(
    #             cls.project_dir, "model_input", raster_name
    #         )

# -*- coding: utf-8 -*-
""" The file has been build for different training routines """
#

import pandas as pd
import seaborn as sns  # SK
from SM.io import read_data, save_model, create_path

from SM.process import (
    create_daily_XY,
    custom_cv_5folds_spatially,
    preprocess,
    preprocess_spatiotempmodel,
    split_data,
    scale_data,
)
from SM.misc import platform_release, check_path
from SM.eval import mean_error, residuals
from SM.cfg import Project
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import matplotlib.pylab as plt
import os
from pathlib import Path


class TrainDailyModel(object):
    def __init__(
        self,
        model,
        method,
        name,
        details,
        project_dir=None,
        training_days=None,
        features=None,
        in_data_name=None,
        in_raster_name=None,
    ):
        self.CWD = os.getcwd()
        self.model = model
        self.method = method
        self.name = name
        self.details = details
        self.uid = self.name + "_" + self.method
        self.platform = platform_release()
        
        self.in_data_name=in_data_name

        self.in_raster_name=in_raster_name,
        self.project_dir=project_dir
        
        # # def additional path
        if self.in_data_name is not None:
            Project.input_data = self.in_data_name
        if self.in_raster_name is not None:
            Project.in_raster_name = self.in_raster_name

        # def paths
        Project.set_project_paths(
            project_dir=self.project_dir, project_type="TrainDailyModel"
        )
        
        self.save_models_path = os.path.join(
            Project.models_path, self.name, self.method
        )
        self.save_figures_path = os.path.join(
            Project.figures_path, self.name, self.method
        )
        self.save_results_path = Project.results_path
        self.features = features


        # read data
        raw_data = read_data()
        self.training_days = training_days
        # set features before data is preprocessed
        Project.set_features(features=features)
        self.processed_data = preprocess(raw_data)
        self.list_unique_days = self.processed_data.Date.unique()

        print(
            f"There are total {len(self.list_unique_days)} unique days and thus same number of models will be trained"
        )

        self.counter = 1

        header = ["Date", "Method", "Name", "Details", "R2_score", "MAE", "RMSE", "ME"]
        self.results = pd.DataFrame(columns=header, index=None)

        self.y_pred_list = []
        self.y_test_list = []

        # check if the path to the saved models already exitst and grabs a new
        # path from user input until it it unique
        check_path(self.save_models_path)

    def fit(self):

        print(
            f"A folder will be created in figures and models folder with the name {self.uid}"
        )

        Path(self.save_models_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_figures_path).mkdir(parents=True, exist_ok=True)

        for each_day in self.list_unique_days:

            day = pd.to_datetime(str(each_day))
            str_day = day.strftime("%Y-%m-%d")

            daily_data = self.processed_data[self.processed_data["Date"] == each_day]

            X_train_scaled, X_test_scaled, y_train, y_test, scaler = create_daily_XY(
                daily_data
            )

            print(
                f"Fitting the model for the day {str_day} --- {self.counter}/{len(self.list_unique_days)}"
            )

            self.model.fit(X_train_scaled, y_train)
            ## Creating custom result files

            y_pred = self.model.predict(X_test_scaled)

            self.y_pred_list.extend(y_pred)
            self.y_test_list.extend(y_test)

            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            me = mean_error(y_true=y_test, y_pred=y_pred)

            self.results.loc[self.counter] = [
                each_day,
                self.method,
                self.name,
                self.details,
                r2,
                mae,
                rmse,
                me,
            ]

            file_name = self.uid + "_" + str_day + "_" + str(self.counter)
            file_path = os.path.join(self.save_models_path, file_name)

            save_model(file_path, self.model, scaler)

            self.counter += 1

        results_path = os.path.join(self.save_results_path, self.uid + ".csv")
        create_path(self.save_results_path)
        self.results.to_csv(results_path, index=False)

        return None

    def scatter_plot(self):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        min_val = min(min(self.y_pred_list), min(self.y_test_list))
        max_val = max(max(self.y_pred_list), max(self.y_test_list))

        min_val = min(min_val, 0)

        ax.scatter(self.y_pred_list, self.y_test_list, color="rebeccapurple", s=1.0)
        ax.plot(
            [min_val, max_val], [min_val, max_val], color="firebrick", linestyle="-"
        )

        ax.set_xlabel("Prediction")

        ax.set_ylabel("Observation")

        scatter_path = os.path.join(self.save_figures_path, self.uid + "_scatter.png")

        plt.savefig(scatter_path)

    def metrics_ts_plot(self):

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8))

        day = self.results["Date"]
        r2_score = self.results["R2_score"]
        mae = self.results["MAE"]
        rmse = self.results["RMSE"]
        me = self.results["ME"]

        ax1.plot(day, rmse, color="rebeccapurple")
        ax1.set_ylabel("RMSE [..]")

        ax2.plot(day, mae, color="rebeccapurple")
        ax2.set_ylabel("MAE [..]")

        ax3.plot(day, r2_score, color="rebeccapurple")
        ax3.set_ylabel("R2 Score [-]")

        ax4.plot(day, me, color="rebeccapurple")
        ax4.set_ylabel("Mean Error [..]")

        ts_path = os.path.join(self.save_figures_path, self.uid + "_metrics_ts.png")

        plt.savefig(ts_path)


class SpatioTempModel(object):
    """A Class to define regression models across time and space."""

    def __init__(
        self,
        estimator,
        method,
        name,
        details,
        project_dir=None,
        features=None,
        in_data_name=None,
        in_raster_name=None,
    ):

        self.CWD = os.getcwd()
        self.model = estimator
        self.method = method
        self.name = name
        self.details = details
        self.uid = self.name + "_" + self.method
        self.platform = platform_release()

        self.project_dir = project_dir
        self.in_data_name = in_data_name
        self.in_raster_name = in_raster_name

        
        # # def additional path
        if self.in_data_name is not None:
            Project.input_data = self.in_data_name
        if self.in_raster_name is not None:
            Project.in_raster_name = self.in_raster_name

        # def paths
        Project.set_project_paths(
            project_dir=self.project_dir, project_type="SpatioTempModel"
        )


        self.save_models_path = os.path.join(
            Project.models_path, self.name, self.method
        )
        self.save_figures_path = os.path.join(
            Project.figures_path, self.name, self.method
        )
        self.save_results_path = Project.results_path
        self.save_residuals_path = Project.residuals_path
        self.save_performance_stats_path = Project.performance_stats_path
        self.save_tuner_results_path = Project.hyperparameters_tuning_path

        # def features
        Project.set_features(features=features)

        # read data
        raw_data = read_data()
        self.processed_data = preprocess_spatiotempmodel(raw_data)

        # initialize placeholders
        self.y_pred_list = []
        self.y_test_list = []

        # check if the path to the saved models already exitst and grabs a new
        # path from user input until it it unique
        check_path(self.save_models_path)

    def train_test(
        self, train_with_cv=False, tuning_hyperparameters=None, splitseed=42
    ):
        """Method to split and scale input data, train and test a model, as well
        as output the results and model.

        Parameter
        ---------
        train_with_cv : boolean, Default: False
            Train model with cross-validation. 'train_with_cv'==True is necessary
            for tuning of hyperparameters defined with 'tuning_hyperparameters'.
        tuning_hyperparameters : dict, Default: None
            Optional hyperparameters for tuning,
            e.g.{"param1": [value1, value2]}.
            If None, no hyperparameter tuning will be performed.
        """

        print(
            f"A folder will be created in figures and models folder with"
            + " the name {self.uid}"
        )

        Path(self.save_models_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_figures_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_results_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_performance_stats_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_tuner_results_path).mkdir(parents=True, exist_ok=True)

        # split data by locations
        X_train, X_test, y_train, y_test = split_data(
            self.processed_data,
            split_type="spatial",
            train_size=0.8,
            random_seed=splitseed,
        )

        aux_data_test = X_test[["Box", "Sensor", "x", "y", "z", "Date", "Date_integer"]]

        # train with cv or without
        if train_with_cv:

            # def cv-folds
            cv_folds = custom_cv_5folds_spatially(X_train)

            # reduce data to relevant features
            X_train = X_train[Project.features_select]
            X_test = X_test[Project.features_select]

            # tune hyperparameters if given
            # if not given, use parameters from self.model to pass to
            # GridSearchCV, however do not tune them
            if tuning_hyperparameters is None:
                tuning_hyperparameters = self.model.get_params()
                # GridSearchCV required hyperparameter values as list
                for key, value in tuning_hyperparameters.items():
                    tuning_hyperparameters[key] = [value]

            # adopt hyperparameter naming for pipeline
            tuning_hyperparameters_new = {}
            for key, value in tuning_hyperparameters.items():
                tuning_hyperparameters_new["model__" + key] = value
            tuning_hyperparameters = tuning_hyperparameters_new
            del tuning_hyperparameters_new

            # def pipeline and grid search
            pipe = Pipeline([("scaler", StandardScaler()), ("model", self.model)])
            tuner = GridSearchCV(
                estimator=pipe,
                param_grid=tuning_hyperparameters,
                refit=True,
                cv=cv_folds,
                scoring="r2",
            )

            # train on training set, predict on test set
            tuner.fit(X_train, y_train)
            y_model_pred = tuner.predict(X_test)

            # write tuning results of all cv-folds to disk
            tuner_results = tuner.cv_results_
            tuner_results_path = os.path.join(
                self.save_tuner_results_path, self.uid + "_tuning.csv"
            )
            pd.DataFrame(tuner_results).to_csv(tuner_results_path, index=False)

            # write tuning result of best tuning result to disk
            tuner_result_best = pd.DataFrame(tuner_results).loc[tuner.best_index_]
            tuner_result_best_path = os.path.join(
                self.save_tuner_results_path, self.uid + "_tuning_best.csv"
            )
            tuner_result_best.to_csv(tuner_result_best_path, index=True, header=None)

            # update model with trained one and get scaler
            self.model = tuner.best_estimator_.named_steps.model
            scaler = tuner.best_estimator_.named_steps.scaler

        else:  # train without cv
            if tuning_hyperparameters:
                print(
                    f"You did define hyperparameters but did not set "
                    + '"train_with_cv" to True.\n'
                    + "Hyperparameter tuning will not be executed."
                )

            # reduce to relevant features
            X_train = X_train[Project.features_select]
            X_test = X_test[Project.features_select]
            # scale data
            X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
            # train
            self.model.fit(X_train_scaled, y_train)
            # test
            y_model_pred = self.model.predict(X_test_scaled)

        # compute model performance stats
        r2 = r2_score(y_test, y_model_pred)
        rmse = mean_squared_error(y_test, y_model_pred, squared=False)
        mae = mean_absolute_error(y_test, y_model_pred)
        me = mean_error(y_true=y_test, y_pred=y_model_pred)
        epsilon = explained_variance_score(y_true=y_test, y_pred=y_model_pred)

        # add results
        header = [
            "Method",
            "Name",
            "Details",
            "R2_score",
            "MAE",
            "RMSE",
            "ME",
            "Exp_variance",
        ]
        self.results = pd.DataFrame(
            [
                dict(
                    zip(
                        header,
                        [
                            self.method,
                            self.name,
                            self.details,
                            r2,
                            mae,
                            rmse,
                            me,
                            epsilon,
                        ],
                    )
                )
            ]
        )

        self.y_pred_list.extend(y_model_pred)
        self.y_test_list.extend(y_test)

        # save model
        file_path = os.path.join(self.save_models_path, self.uid)
        save_model(file_path, self.model, scaler)

        # write results into performance_stats directory
        performance_stats_path = os.path.join(
            self.save_performance_stats_path, self.uid + ".csv"
        )
        create_path(self.save_performance_stats_path)
        self.results.to_csv(performance_stats_path, index=False)

        # compute residuals
        res = residuals(y_true=self.y_test_list, y_pred=self.y_pred_list)
        res_df = pd.DataFrame(
            res, columns=["y_true", "y_pred", "residuals"], index=None
        )
        # merge residuals dataframe with coordinates and dates of measurement points
        # this will help in plotting maps, scatter plots with time, computing f statistic etc.
        self.residuals = pd.concat(
            (aux_data_test.reset_index(drop=True), res_df), axis=1
        )

        # write residuals
        residuals_path = os.path.join(self.save_residuals_path, self.uid + ".csv")
        create_path(self.save_residuals_path)
        self.residuals.to_csv(residuals_path, index=False)

        # feature importance
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        self.r_train = permutation_importance(
            self.model, X_train_scaled, y_train, n_repeats=15, random_state=42  # 0,
        )
        self.r_test = permutation_importance(
            self.model, X_test_scaled, y_test, n_repeats=15, random_state=42  # 0,
        )

    #        row = []
    #        for i in self.r_train.importances_mean.argsort()[::-1]:
    #            if self.r_train.importances_mean[i] - 2 * self.r_train.importances_std[i] > 0: # what happens here? (JS)
    #                row.append([self.uid, self.details,Project.features_select[i], self.r_train.importances_mean[i], self.r_train.importances_std[i]])
    #        self.imp_features = pd.DataFrame.from_records(row, columns = ["uid","details","feature", "mean", "sdev"])

    def scatter_plot(self):
        """Create and save a scatter plot of soil moisture observations versus
        predictions."""

        # define plot specs
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        # get min max values for setting the axis ranges
        min_val = min(min(self.y_pred_list), min(self.y_test_list))
        max_val = max(max(self.y_pred_list), max(self.y_test_list))
        min_val = min(min_val, 0)

        # plot
        ax.scatter(self.y_test_list, self.y_pred_list, color="rebeccapurple", s=1.0)
        ax.plot(
            [min_val, max_val], [min_val, max_val], color="firebrick", linestyle="-"
        )

        ax.set_xlabel("Observation")
        ax.set_ylabel("Prediction")

        # save
        scatter_path = os.path.join(self.save_figures_path, self.uid + "_scatter.png")
        plt.savefig(scatter_path)

    def box_plot(self):
        """Calculate residuals and save them as a box plot for the model run"""

        residuals_data = self.residuals

        # define plot specs
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))

        # plot
        sns.boxplot(residuals_data["residuals"], ax=axes, color="orange")

        axes.set_xlabel("Residuals")
        axes.set_ylabel("")
        axes.set_title("Residuals for model: " + self.uid)

        # save
        boxplot_path = os.path.join(
            self.save_figures_path, self.uid + "_residuals_boxplot.png"
        )
        plt.savefig(boxplot_path)

    def box_plot_feature_importance(self, feature_list=None):
        """Plots distribution of feature importance of training and
        testing sets, as calculated by permutation importance function
        provided by sklearn.

        Parameter:
        self: class object
        feature_list: List of string variables as defined in the run script.
        If using the same as the base feature list, then the user has to make sure
        that they are identical to each other. Default option takes the features
        from the Project configuration.

        Returns:
        matplotlib boxplot.

        """
        if feature_list is None:
            features_toplot = Project.features_select
        else:
            features_toplot = feature_list
        print("Plotting the following features and their importances: ")
        print(features_toplot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
        plt.suptitle("Feature importance")

        ax1.set_title("Training set")
        ax2.set_title("Test set")
        ax1.boxplot(self.r_train.importances.T, vert=False, labels=features_toplot)
        ax2.boxplot(self.r_test.importances.T, vert=False, labels=features_toplot)
        fig.tight_layout()
        # save
        boxplot_path = os.path.join(
            self.save_figures_path, self.uid + "_feature_importance_boxplot.png"
        )
        plt.savefig(boxplot_path)

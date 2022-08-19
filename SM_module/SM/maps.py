# -*- coding: utf-8 -*-
""" The file has been build for creating beautiful maps! """
#

import os
from SM.io import read_raster, load_model, read_data
from SM.process import (
    preprocess_raster,
    preprocess_raster_spatiotempmodel,
    raster_select_features,
    create_gdf_from_df,
)
from SM.cfg import Project
from SM.training import SpatioTempModel

import pandas as pd

import matplotlib.pylab as plt
import matplotlib as mpl
import geopandas as gpd
import fnmatch
from pathlib import Path


class SpatialMap(object):
    def __init__(self, name, method, date, csv_file=None, plot_sensor_locs=True):

        self.name = name
        self.method = method
        self.uid = self.name + "_" + self.method
        self.csv_file = csv_file
        self.date = date
        self.save_models_path = os.path.join(
            Project.models_path, self.name, self.method
        )
        self.save_figures_path = os.path.join(
            Project.figures_path, self.name, self.method
        )
        self.save_results_path = os.path.join(Project.figures_path)
        self.pred_gdf = None
        self.plot_sensor_locs = plot_sensor_locs

        Path(self.save_models_path).mkdir(parents=True, exist_ok=True)
        Path(self.save_figures_path).mkdir(parents=True, exist_ok=True)

        if self.csv_file is not None:
            print("The specified csv file will be taken for map creation.")
            self._create_gdf_from_csv()
        else:
            if Project.project_type == "SpatioTempModel":
                self._predict_spatiotempmodel()
            else:
                self._predict()

        if self.plot_sensor_locs == True:
            try:
                self.daily_gdf = read_data()
            except FileNotFoundError:
                self.plot_sensor_locs = False
                print("Found no files with sensor locations.")

    def plot_maps(self, grid=True):
        """Plot predicted raster data as maps and save as PNG.

        Parameter
        ---------
        grid : boolean, Default: True
            Set to True to plot the raster grid lines.
        """

        # to plot sensor locatoins, load training_test data
        if self.plot_sensor_locs == True:

            daily_gdf = self.daily_gdf

            if not isinstance(daily_gdf, gpd.GeoDataFrame):
                daily_gdf = create_gdf_from_df(
                    daily_gdf, x="UTMWGS84_E[m]", y="UTMWGS84_N[m]"
                )

            if "Date" in daily_gdf.columns:
                try:
                    # if model is of class SpatioTempModel
                    daily_gdf = daily_gdf[daily_gdf["Date"] == self.date]
                except:
                    # if model is of class TrainDailyModel
                    daily_gdf = daily_gdf[
                        daily_gdf["Date"] == pd.to_datetime(self.date)
                    ]

        # load predicted raster data
        pred_gdf = self.pred_gdf
        if not isinstance(pred_gdf, gpd.GeoDataFrame):
            pred_gdf = create_gdf_from_df(pred_gdf)

        print("######")
        print(pred_gdf.columns)

        # check depth levels to create plot grid
        depth_list = pred_gdf["z"].unique()
        n_depth = len(depth_list)

        # create plot grid and set color scheme
        fig, axes = plt.subplots(
            1, n_depth, figsize=(3 * n_depth, 6), sharey=True, sharex=True
        )
        # cmap = 'cviridis_r'
        cmap = mpl.cm.get_cmap("viridis")
        cmap = cmap.reversed()
        norm = mpl.colors.Normalize(vmin=0.01, vmax=0.45)
        cmap.set_under("white")
        cmap.set_over("black")
        fmt = lambda x, pos: "{:.0f}".format(x)

        # plot depth specific maps
        for i in range(n_depth):

            gdf_d = pred_gdf[pred_gdf["z"] == depth_list[i]]

            gdf_d_plot = gdf_d.plot(ax=axes[i], column="pred", cmap=cmap, norm=norm)
            if self.plot_sensor_locs == True:
                daily_gdf_d = daily_gdf[daily_gdf["Depth_m"] == depth_list[i]]
                daily_gdf_d.plot(
                    ax=axes[i],
                    column="Soil_moisture",
                    cmap=cmap,
                    norm=norm,
                    edgecolor="k",
                )

            axes[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            axes[i].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            axes[i].set_title("Depth [m]: " + str(depth_list[i]), pad=20)
            if grid == True:
                axes[i].grid(color="grey", linestyle="--")
            collection = gdf_d_plot.collections[0]

        cbar = plt.colorbar(collection, ax=axes[-1], extend="both")
        cbar.ax.set_ylabel("Soil moisture [-]", fontsize=14)

        # save maps as png
        f_path = os.path.join(
            self.save_figures_path, self.uid + "_" + self.date + "_3d.png"
        )
        plt.savefig(f_path)

    def _find_date_model_fname(self, date, model_path):

        list_files = os.listdir(model_path)

        file_name = None

        for each_file in list_files:
            if fnmatch.fnmatch(each_file, "*_" + date + "*_model.sav"):
                file_name = each_file.split(".")[0][:-6]

        if file_name == None:
            raise FileNotFoundError(
                "There is no saved model with the specified name. \nTrain a model first or specify a csv file."
            )
        date = file_name.split("_")[2]

        return date, file_name

    def _predict(self):

        raster_data = read_raster()
        raster_data = raster_data[raster_data["mask"] == False]
        processed_raster_data = preprocess_raster(raster_data)

        date, fname = self._find_date_model_fname(self.date, self.save_models_path)

        fpath = os.path.join(self.save_models_path, fname)

        model, scaler = load_model(fpath)

        data_scaled = scaler.transform(processed_raster_data.values)

        sm_pred = model.predict(data_scaled)

        raster_data["pred"] = sm_pred

        raster_data = raster_data.rename(columns={"z": "Depth_m"})

        self.pred_gdf = raster_data
        # self.date = date

        # return self.date, self.pred_gdf
        return self.pred_gdf

    def _predict_spatiotempmodel(self):
        """Predict a map for a specific time stamp for a model of class
        SpatioTempModel.

        Returns
        -------
        pandas.DataFrame
            Spatial map, columns ("x", "y", "z", "mask", "pred").
        """
        # load and preprocess raster data
        raster_data = read_raster()
        raster_data = raster_data[raster_data["mask"] == False]
        processed_raster_data = preprocess_raster_spatiotempmodel(raster_data)
        feature_selected_raster_data = raster_select_features(
            processed_raster_data, self.date
        )
        # load regression model
        fpath = os.path.join(self.save_models_path, self.uid)
        model, scaler = load_model(fpath)

        # scale, predict
        data_scaled = scaler.transform(feature_selected_raster_data.values)
        sm_pred = model.predict(data_scaled)

        # output
        pred_raster = processed_raster_data[["x", "y", "z", "mask"]]
        pred_raster = pred_raster.assign(pred=sm_pred)
        self.pred_gdf = pred_raster

        # self.pred_gdf should look like this
        #            x           y     z     mask      pred
        # 0      641952.55  5724795.24  0.05  False  0.229296
        return self.pred_gdf

    def _create_gdf_from_csv(self):

        df = pd.read_csv(self.csv_file)
        df = df[df["mask"] == False]

        self.pred_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

        return self.pred_gdf

    def save_csv(self, columns=["x", "y", "z", "pred", "mask"]):
        """Save predicted maps as csv file."""

        pred_gdf_export = self.pred_gdf[columns]
        pred_gdf_export.to_csv(
            os.path.join(self.save_figures_path, self.uid + "_" + self.date + "_3d.csv")
        )

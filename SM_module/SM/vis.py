#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# Created by : Pia Ebeling
# Created on : Mon May 31 14:08:16 2021
# ======================================================================
# __author__ = Pia Ebeling
# __copyright__ = Copyright (c) 2021, Pia Ebeling, Project Soil Moisture
# __credits__ = [Pia Ebeling]
# __license__ = MIT
# __version__ = 0.0.1
# __maintainer__ = Pia Ebeling
# __email__ = pia.ebeling@ufz.de
# __status__ = development
# ======================================================================
""" The file provides functions to compare different machine learning models and/or algortihms """

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import geopandas as gpd
import matplotlib.dates as mdates
import os
from sklearn.metrics import r2_score, mean_squared_error

import seaborn as sns  # SK

# import from SM module
from SM.cfg import Project
from SM.process import create_gdf_from_df

def compare_perform_boxplots(filenames, measures, project_dir, pltname):
    """
    This function generates boxplots from performance measures from different models (files in project_dir/performance_stats).
    The created figure is saved with the extension given in under project_dir/figures/comp_perform_boxplots_+pltname.

    Parameters
    ----------
    filenames : list of string
        List of model names to be compared. If "all", all result files from results folder will be shown (maximum 10)
    measures : list of string
        List of performance measures to be compared.
    project_dir : string
        Project directory with subdirectories "results" and "figures".
    pltname : string
        Name extension to use for filename of the figure.

    Returns
    -------
    None.

    """
    print(filenames)
    print(measures)

    # Load model results and concatenate
    path = os.path.join(project_dir, "performance_stats")
    if filenames == "all":
        all_files = os.listdir(path)
        filenames = all_files[0:11]  # maximum files plotted
    print(filenames)

    # read files and concatenate
    # cols = ["Method","Name","Details"] + measures
    li = []
    for file in filenames:
        df = pd.read_csv(os.path.join(path, file+'.csv'), index_col=None, header=0)
        # check columns of files
        print(file)
        print(df[0:2])
        # append file to list
        li.append(df)
    results_all = pd.concat(li, axis=0, ignore_index=True)
    print(results_all.columns)

    # prepare subplots for each measure with all models
    nrows, ncols = 1, len(measures)
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "grey",
        "c",
        "m",
        "y",
        "k",
        "w",
    ]
    methods = set(results_all["Method"])
    methods_short = methods  # abbreviations can be adapted here

    # plot subplots
    fig = plt.figure(figsize=(15, 4))
    plt.subplots_adjust(hspace=0.5)
    # fig.suptitle("3D models", fontsize=14)
    for i, measure in enumerate(measures):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        if measure == "R2_score":
            ax.axhline(y=1, color="blue", linestyle="--", linewidth=1)
            ax.axhline(y=0, color="grey", linestyle="--", linewidth=1)
        else:
            ax.set_ylim([-0.01, max(results_all[measure]) * 1.01])
            ax.axhline(y=0, color="blue", linestyle="--", linewidth=1)
        print(i, measure, ax)
        bp = ax.boxplot(
            [
                results_all.loc[results_all.Method == method][measure]
                for method in methods_short
            ],
            patch_artist=True,
        )
        ax.set_title(measure, fontsize=14)
        ax.set_xticklabels(methods_short, rotation=0, fontsize=13)
        ax.tick_params(labelsize=13)
        # fill with colors
        for patch, color in zip(bp["boxes"], colors):
            print(patch, color)
            patch.set(facecolor=color, alpha=0.8)
    filename = os.path.join(
        project_dir, "figures/comp_perform_boxplots_" + pltname + ".png"
    )
    fig.savefig(filename, dpi=300)

    return None


def plot_boxplot(data, ycolumn, xcolumn):
    """Function to plot boxplots depicting residuals for
    multiple models in one figure. The residuals are to be loaded
    independently from a csv file.
    
    Parameter
    ---------
    data: pandas dataframe with columns unique ID (model.uid) and residuals.
    ycolumn: string, specify column header that houses the residuals dataset
    xcolumn: string, specify column header that houses the categorical variable.
    In our case,this is the UID.
    """
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    # plot
    sns.boxplot(x=xcolumn, y=ycolumn, data=data, hue=xcolumn, ax=axes)
    axes.set_xlabel(xcolumn)
    axes.set_ylabel(ycolumn)
    axes.set_title("Boxplots for " + ycolumn)

    return fig


def f_heatmap(data, ycolumn):
    """
    Function to plot heatmap of a value of a pair of models with a diverging color palette.
    For example, F-statistic.
    
    Parameter
    --------
    data: Pandas dataframe with at least 2 columns titled UID_1 and UID_2.
    ycolumn: Column header for values to be plotted in the heatmap.
    
    Returns
    -------
    None
    """

    import seaborn as sns

    df_pivot = data.pivot(index="UID_1", columns="UID_2", values=ycolumn)
    sns.heatmap(df_pivot, cmap="vlag")

    return None


def temporal_scatter(df, ycolumn, xcolumn, huecolumn, project_dir):
    """
    Function to plot scatterplot of 2 variables with color varying
    according to a variable, with fixed 3 columns and number of rows dependent on
    total number of models.
    
    Parameter
    --------
    df: Pandas dataframe with at least 2 columns titled UID_1 and UID_2.
    ycolumn: Column header for values to be plotted in the scatterplot.
    xcolumn: Column header for values in the X axis.
    huecolumn: Column header for values for variation of colour.
    
    Returns
    -------
    Scatter plot.
    """

    Project.set_project_paths(project_dir=project_dir, project_type="SpatioTempModel")

    Bluescmap = mpl.cm.Blues(np.linspace(0, 1, 40))

    uid_list = df.UID.unique().tolist()
    grid_cols = 2
    grid_rows = int(len(uid_list) / grid_cols)
    if grid_rows==0:
        grid_rows=1
    color_map = mpl.colors.ListedColormap(Bluescmap[10:, :-1])
    fig, axes = plt.subplots(ncols=grid_cols, nrows=grid_rows, sharex=True, sharey=True,
                             figsize = (10,8))
    for u in uid_list:
        data = df[df.UID == u]
        idx = uid_list.index(u)
        ax = axes.flat[idx]
        ax.scatter(
            x=data[xcolumn], y=data[ycolumn], c=data[huecolumn], cmap=color_map, s=20
        )
        ax.set_title(u)
        if idx >= (grid_cols * (grid_rows - 1)):
            ax.set_xlabel(xcolumn)
        else:
            ax.set_xlabel("")
        if idx % grid_cols == 0:
            ax.set_ylabel(ycolumn)
        else:
            ax.set_ylabel("")
        if xcolumn == "Date":
            ax.set_xticks(
                ["2012-09-16", "2013-01-01", "2013-04-01", "2013-07-01", "2013-10-01"]
            )
            ax.set_xticklabels(
                ("16/09", "01/01", "04/01", "01/07", "01/10"), rotation=60
            )
    min_val = mpatches.Patch(
        color=Bluescmap[10, :-1], label=str(np.round(np.min(df[huecolumn]), 2))
    )  # , alpha = 0.5)
    mid_val = mpatches.Patch(
        color=Bluescmap[int(np.shape(Bluescmap)[0] / 2 + 5)],
        label=str(np.round(np.mean(df[huecolumn]), 2)),
    )  # , alpha = 0.5)
    max_val = mpatches.Patch(
        color=Bluescmap[-1, :-1], label=str(np.round(np.max(df[huecolumn]), 2))
    )  # , alpha = 0.5)

    dotlist = [min_val, mid_val, max_val]
    plt.legend(handles=dotlist, bbox_to_anchor=(0.1, -0.5), ncol=3, title=huecolumn)

    # save
    scatter_path = os.path.join(Project.figures_path, "residuals_scatter_with_time.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")

    return None


def mean_r2_rmse_cv(g):
    """
    This function computes mean, R2, RMSE and CV of the pandas dataframe in compare_perform_residualmaps()
    The function is used for aggregating with groupby().
    
    Parameters
    ----------
    g: Pandas dataframe 

    Returns
    -------
    Pandas series.

    """
    x = np.mean(g['x'])
    y = np.mean(g['y'])
    mean = np.mean(g['residuals'])
    r2 = r2_score( g['y_true'], g['y_pred'] )
    rmse = mean_squared_error(g['y_true'], g['y_pred'], squared=False)
    cv = np.std(g['residuals'], ddof=1) / np.mean(g['residuals'])
    return pd.Series(dict(x=x, y=y, mean = mean, r2 = r2, rmse = rmse, cv = cv ))


def compare_perform_residualmaps(data, project_dir, measure="mean"):
    """
    This function generates spatial maps of residuals from different models (files in project_dir/results).
    The created figure is saved with the extension given in under project_dir/figures/compare_perform_residualmaps+measure.png

    Parameters
    ----------
    data: Pandas dataframe 
    measure : string
        Performance measures to be plotted:
            "mean": Mean residuals per box (over time, depths and sensors)
            "r2": R2
            "rmse": Root mean squared error
            "cv": Coefficient of variation of residuals
    project_dir : string
        Project directory with subdirectories "results" and "figures".
        
    Returns
    -------
    None.

    """
    
    # Number of models in "consolidated_residuals.csv"
    models = data.UID.unique()
    N = len(data.UID.unique())

    # Build mean, R2, RMSE and CV over residuals of a specific box
    data_mean = data.groupby(["UID", "Box"]).apply(mean_r2_rmse_cv).reset_index()
    
    # convert Pandas dataframe to GeoDataFrame
    if not isinstance(data_mean, gpd.GeoDataFrame):
                data_mean = create_gdf_from_df(
                    data_mean, x="x", y="y"
                )
    
    # Legend lables
    if measure == "mean":
        legendlabel = "Mean residuals [-]"
        cmap = mpl.cm.get_cmap("seismic")
    if measure == "r2":
        legendlabel = "R2 [-]"
        cmap = mpl.cm.get_cmap("BuGn")
    if measure == "rmse":
        legendlabel = "RMSE [-]" 
        cmap = mpl.cm.get_cmap("YlOrRd")
    if measure == "cv":
        legendlabel = "Coefficient of variation of residuals [-]"
        cmap = mpl.cm.get_cmap("seismic")
    
    fmt = lambda x, pos: "{:.0f}".format(x)
    
    fig, axes = plt.subplots(
        1, N, figsize=(2 * N, 6), sharey=True, sharex=True
    )
    for i in range(N):
        data_mean_plot = data_mean.plot(
            ax=axes[i],
            column=measure,
            cmap=cmap,
            edgecolor="k"
            )
        axes[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        axes[i].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        axes[i].set_title(str(models[i]), pad=20)
        collection = data_mean_plot.collections[0]
    
    cbar = plt.colorbar(collection, ax=axes[-1], extend="both")
    cbar.ax.set_ylabel(legendlabel, fontsize=14)
    
    # save maps as png
    output_file = "figures/compare_perform_residualmaps" + measure + ".png"
    print("Saving maps of all models in: " + output_file)
    f_path = os.path.join(
        project_dir, output_file 
        )
    plt.savefig(f_path)    
    return None   

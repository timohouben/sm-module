# -*- coding: utf-8 -*-
""" The file has been build for pre and post processing model results """
#

from SM.io import read_data
from SM.cfg import Project
import fnmatch

# from SM.training import SpatioTempModel
import datetime as dt
import pandas as pd
import numpy as np

import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os


def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    data["aspect_cos"] = np.cos(convert_rad(data["aspect"]))
    data["aspect_sin"] = np.sin(convert_rad(data["aspect"]))

    # We prepare x and y as feature, they might not be used but just are prepared
    
    data.geometry
    data["x"] = data.geometry.x
    data["y"] = data.geometry.y

    ## We select the features here
    print(Project.features_select)
    selected_feature_data = data[Project.features_select]

    return selected_feature_data


def preprocess_spatiotempmodel(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for models of class SpatioTempModel.

    Paramters
    ---------
    data : pandas.DataFrame
        Input data with columns:
          [;Date_integer;Date;Box;Sensor;Soil_moisture;P_mm;PET_mm;Depth_m;Region;
          UTMWGS84_E[m];UTMWGS84_N[m];Elevation[m];Silt%;Clay%;Sand%;Porosity%;
          dist_to_creek;twi;relief_1;relief_2;relief_3;hillshade;rugg_idx;slope;
          aspect;ele_dem;Temp]
    Returns
    -------
    pandas.DataFrame

    """

    data = (
        data.assign(
            x=data["UTMWGS84_E[m]"],
            y=data["UTMWGS84_N[m]"],
            z=data["Depth_m"],
            dayofyear=pd.to_datetime(data["Date"]).dt.dayofyear,
        )
        .drop(columns=["UTMWGS84_E[m]", "UTMWGS84_N[m]", "Depth_m", "Region"])
        .dropna()
    )

    # decode yeardate
    data["dayofyear_sin"] = np.sin(convert_rad(data["dayofyear"] * 360 / 365.25))
    data["dayofyear_cos"] = np.cos(convert_rad(data["dayofyear"] * 360 / 365.25))

    # decode aspect
    data["aspect_cos"] = np.cos(convert_rad(data["aspect"]))
    data["aspect_sin"] = np.sin(convert_rad(data["aspect"]))

    # subset to base columns (important for subsequent processing) and features
    columns = [
        "Date",
        "Date_integer",
        "Box",
        "Sensor",
        "x",
        "y",
        "z",
        "Soil_moisture",
    ] + Project.features_select
    # reordered_columns = columns + (data.columns.drop(columns).tolist())
    data = data[columns]
    data = data.loc[:, ~data.columns.duplicated()]

    return data


def convert_rad(d: float) -> float:
    r = d * np.pi / 180.0
    return r


def create_daily_XY(daily_data):

    X = daily_data.drop(["Soil_moisture", "Date"], axis=1).values
    y = daily_data.Soil_moisture.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=None
    )

    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def split_data(X, train_size, split_type, random_seed=42):
    """
    Wrapper to split data into a training set and a test set based on: 1) locations (boxes) or 2) randomly.

    Parameters
    ---------
    X : pandas.Dataframe
        Data set to split: Measurements of soil moisture and all features.
    train_size : float
        Proportion of boxes ("spatial") or data ("random") that is used for the training set.
    split_type : string
        two options: either  "spatial" for splitting by location (boxes) or "random" for totaly random split in space and time.
    random_seed : integer default: 42
        Value of random seed for creating the random split, useful for creating different splits.

    Returns
    -------
    X_train: pandas.Dataframe
        Dataset containing features for training.
    X_test: pandas.Dataframe
        Dataset containing features for testing.
    y_train: pandas.Series
        Soil moisture values of training set.
    y_test: pandas.Series
        Soil moisture values of test set.
    """

    if split_type == "spatial":
        train_set, test_set = shuffle_split_data_spatially(X, train_size, random_seed)

    if split_type == "random":
        train_set, test_set = shuffle_split_data_randomly(X, train_size, random_seed)

    X_train = train_set.drop(["Soil_moisture"], axis=1)
    y_train = train_set.Soil_moisture
    X_test = test_set.drop(["Soil_moisture"], axis=1)
    y_test = test_set.Soil_moisture

    return X_train, X_test, y_train, y_test


def shuffle_split_data_spatially(X, train_size, random_seed):
    """
    Split data into a training set and test set based on locations (boxes).
    All rows with the same value for 'box' will be either go into training or test set.

    Parameters
    ---------
    X : Pandas Dataframe
        Dataset to split: Measurements of soil moisture and all features
    train_size : float
        Proportion of boxes used for the training set (not measurements!)
    random_seed : integer
        Value of random seed for creating the random split

    Returns
    -------
    train_set : Pandas Dataframe
        Dataset containing the data for training
    test_set : Pandas Dataframe
        Dataset containing the data for testing
    """

    np.random.seed(random_seed)  # set random seed for reproducibility

    # split available boxes accordong to train_size
    boxes = X.Box.unique()
    arr_rand = np.random.rand(boxes.shape[0])
    split_boxes = arr_rand < np.percentile(arr_rand, train_size * 100)
    train_boxes = boxes[split_boxes]
    test_boxes = boxes[~split_boxes]

    # split entire data set
    train_set = X[X.Box.isin(train_boxes)]
    test_set = X[X.Box.isin(test_boxes)]

    print(
        "Splitting spatially:\nTrain set:",
        len(train_set),
        " measurements of",
        len(train_boxes),
        " boxes \n Test set:",
        len(test_set),
        " measurements of",
        len(test_boxes),
        " boxes",
    )
    return train_set, test_set


def shuffle_split_data_randomly(X, train_size, random_seed):
    """
    Split data into a training set and test set randomly in space and time.

    Parameters
    ---------
    X : Pandas Dataframe
        Dataset to split: Measurements of soil moisture and all features
    train_size : float
        Proportion of measurements used for the training set
    random_seed : integer
        Value of random seed for creating the random split

    Returns
    -------
    train_set : Pandas Dataframe
        Dataset containing the data for training
    test_set : Pandas Dataframe
        Dataset containing the data for testing
    """

    np.random.seed(random_seed)  # set random seed for reproducibility

    # split available measurements accordong to train_size
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, train_size * 100)

    train_set = X[split]
    test_set = X[~split]

    print(
        "Splitting randomly: \nTrain set:",
        len(train_set),
        " measurements \n",
        "Test set:",
        len(test_set),
        " measurements",
    )
    return train_set, test_set


def custom_cv_5folds_spatially(X, random_seed=42):
    """
    Creates a generator with indices of spatial split for 5-fold cross validation

    Parameters
    ---------
    X : pandas.Dataframe
        Data set to split: Measurements of soil moisture and all features. Should include feature "Box"
    random_seed : integer default: 42
        Value of random seed for creating the random split, useful for creating different splits.

    Yields
    -------
    generator object with np.arrays(idx_training, idx_testing)
        containing the indices for training and test sets of the five folds
    """

    np.random.seed(random_seed)

    # shuffle boxes to create a random split
    boxes = X.Box.unique()
    np.random.shuffle(boxes)
    # split available boxes into five parts
    split_boxes = np.array_split(boxes, 5)
    # reset index to an complete integer index
    X = X.reset_index(drop=True)
    # create generator with indices
    i = 0
    while i < 5:
        # split entire data set
        idx_test = X.index.values[X.Box.isin(split_boxes[i])]
        idx_train = X.index.values[~X.Box.isin(split_boxes[i])]
        yield idx_train, idx_test
        i += 1


def scale_data(X_train, X_test):
    """Scale training and testing data with StandardScaler().

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training data.
    X_test : pandas.DataFrame
        Test data.

    Return
    ------
    Tuple of scaled data : pandas.DataFrame, scaler
    """
    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train), scaler.transform(X_test), scaler)


def preprocess_raster(raster_data):

    raster_data["aspect_cos"] = np.cos(convert_rad(raster_data["aspect"]))
    raster_data["aspect_sin"] = np.sin(convert_rad(raster_data["aspect"]))

    # We prepare x and y as feature, they might not be used but just are prepared

    raster_data["x"] = raster_data.geometry.x
    raster_data["y"] = raster_data.geometry.y

    ## Remove No values data
    raster_data = raster_data[raster_data["mask"] == False]

    x = Project.features_select

    if "Date" in x:
        x.remove("Date")
    if "Soil_moisture" in x:
        x.remove("Soil_moisture")

    x = [word if word != "Depth_m" else "z" for word in x]

    raster_data = raster_data[x]

    return raster_data


def preprocess_raster_spatiotempmodel(raster_data):
    """Preprocess raw raster data.

    Parameter
    ---------
    raster_data : pandas.DataFrame
        Raw raster data including columns: "coord_x", "coord_y", "z", "mask" and
        features columns.

    Returns
    -------
    pandas.DataFrame

    """

    ## Remove No values data
    raster_data = raster_data[raster_data["mask"] == False]

    raster_data["aspect_cos"] = np.cos(convert_rad(raster_data["aspect"]))
    raster_data["aspect_sin"] = np.sin(convert_rad(raster_data["aspect"]))

    raster_data = (
        raster_data.assign(
            aspect_cos=np.cos(convert_rad(raster_data["aspect"])),
            aspect_sin=np.sin(convert_rad(raster_data["aspect"])),
            x=raster_data["coord_x"],
            y=raster_data["coord_y"],
        )
        .drop(columns=["coord_x", "coord_y"])
        .dropna()
    )
    # reorder
    columns = ["x", "y", "z", "mask"]
    reordered_columns = columns + (raster_data.columns.drop(columns).tolist())
    raster_data = raster_data[reordered_columns]

    return raster_data


def raster_select_features(raster_data, date):
    """Subset raster data to feature columns and, in case, get temporal
    features from the training data.

    Parameters
    ----------
    raster_data : pandas.DataFrame
        Data of the raster with columns.
    date : str
        Date to predict the map for ("yyyy-mm-dd").

    Returns
    -------
    raster_data : pandas.DataFrame
        Subsetted data of the raster.
    """

    # select features
    features = Project.features_select

    # remove unused stuff
    if "Date" in features:
        features.remove("Date")
    if "Soil_moisture" in features:
        features.remove("Soil_moisture")
    features = [word if word != "Depth_m" else "z" for word in features]

    # get temporal features that are not stored in raster_data
    feats_in_raster = [x for x in features for y in raster_data.columns if x == y]
    feats_not_in_raster = [x for x in features if x not in feats_in_raster]

    # get feature values at specific date from traning_testing data
    X_traintest = read_data()
    X_traintest = preprocess_spatiotempmodel(X_traintest)
    X_feats = X_traintest.loc[
        X_traintest["Date"] == date, feats_not_in_raster
    ].drop_duplicates()
    if len(X_feats) == 0:
        raise ValueError("The provided Date is not in the data.")

    # add to raster
    for i, value in enumerate(X_feats.values[0]):
        raster_data[X_feats.columns[i]] = value

    raster_data = raster_data[features]

    return raster_data


def create_gdf_from_df(df, x="x", y="y"):
    """Create a Geopandas.GeoDataFrame out of a pandas.DataFrame.

    Paramters
    ---------
    df : pandas.DataFrame
        Has to have columns ["x", "y", ...].
    x : str, Default: "x"
        Name of the columns for x - coordinates.
    y : str, Default: "y"
        Name of the columns for y - coordinates.

    Returns
    -------
    gdf : geopands.GeoDataFrame
    """
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[x], df[y]), crs=Project.crs
    )
    return gdf


def merge_csv_files(directory, output_file):
    """Function to merge multiple csv files in a directory.
    Load data in each csv file.
    Merge the data together with the a unique identifier (filename) in a new column "UID".
    Save the file to be loaded independently.

    Parameter
    ---------
    directory: string, parent directory under which you want to search for files/data
    Note that all subdirectories will be checked so be sure to identify a suitable project directory.
    output_file: string, complete path with file name (csv) in which residuals are to be stored

    Returns
    ---------
    Merged pandas dataframe.
    """
    print("Identified models and associated data: ")

    list_files = os.listdir(directory)
    i = 0
    for each_file in list_files:
        if fnmatch.fnmatch(each_file, "*.csv"):
            file_name = each_file
            uid = file_name[:-4]
            print(uid)
            data = pd.read_csv(os.path.join(directory, each_file))
            uid_df = pd.Series([uid] * len(data), name="UID")
            residuals_df = pd.concat([uid_df, data], axis=1)
            if i == 0:
                residuals_results = residuals_df
            else:
                residuals_results = residuals_results.append(
                    residuals_df, ignore_index=True
                )
            i = i + 1

    print("Saving residuals of all models to: " + output_file)
    residuals_results.to_csv(output_file, index=False)

    return residuals_results

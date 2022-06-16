#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# Created by : Mohit Anand
# Created on : On Sat Jan 16 2021 at 00:10:48
# ======================================================================
# __author__ = Mohit Anand
# __copyright__ = Copyright (c) 2021, Mohit Anand, Soil Moisture Project
# __credits__ = [Mohit Anand,]
# __license__ = MIT
# __version__ = 0.0.1
# __maintainer__ = Mohit Anand
# __email__ = mohit.anand@ufz.de
# __status__ = development
# ======================================================================
""" The file has been build for all input output transfer of data """
#

import pickle
import pandas as pd
import os

from SM.cfg import Project


def read_data() -> pd.DataFrame:
    """Read model input data based on type of file.
    
    Returns
    -------
    pandas.DataFrame
    
    """
    try:
        with open(Project.data_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            'Please drop the model input data in the "model_input" directory in your project directory, i.e. under '
            + Project.project_dir
            + "/model_input"
        )
    except:
        data = pd.read_csv(Project.data_path, sep=";", index_col=0, na_values="na")

    return data


def read_raster():
    """Read raster input data based on type of file.
    
    Returns
    -------
    pandas.DataFrame
    
    """
    try:
        with open(Project.raster_path, "rb") as f:
            data = pickle.load(f)
    except:
        data = pd.read_csv(Project.raster_path, sep=",")

    return data


def save_model(fname, model, scaler):
    f_model = fname + "_model.sav"
    f_scaler = fname + "_scaler.sav"

    pickle.dump(model, open(f_model, "wb"))
    pickle.dump(scaler, open(f_scaler, "wb"))


def load_model(fname):
    f_model = fname + "_model.sav"
    f_scaler = fname + "_scaler.sav"

    model = pickle.load(open(f_model, "rb"))
    scaler = pickle.load(open(f_scaler, "rb"))
    return model, scaler


def create_path(fname):
    if not os.path.exists(fname):
        os.mkdir(fname)

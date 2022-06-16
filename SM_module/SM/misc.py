#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# Created by : Mohit Anand
# Created on : On Sat Jan 16 2021 at 15:02:21
# ======================================================================
# __author__ = Mohit Anand
# __copyright__ = Copyright (c) 2021, Mohit Anand, Soil Moisture Project
# __credits__ = [Mohit Anand, Timo Houben]
# __license__ = MIT
# __version__ = 0.0.1
# __maintainer__ = Mohit Anand
# __email__ = mohit.anand@ufz.de
# __status__ = development
# ======================================================================
""" The file has been build for miscellaneous functions """
#

# from SM.cfg import models_path, figures_path, results_path
import shutil
import os
import fnmatch


def platform_release():
    """
    Returns platform release number.
    This needs to be updated manually if the release changes. This is not unique
    for EVE and can have same return if executed on another system with same
    release number.

    Returns
    -------
    string
        'eve' if running on UFZ EVE
        'other' if running on other systems
    """
    import platform
                            
    if platform.release() == "3.10.0-1160.15.2.el7.x86_64":
        return "eve"
    else:
        return "local"


## Need to change this a bit
def clear_results(name, method):
    """ 
    Deletes all model output in current working directory.
    Execute from run script. 
    
    Parameters:
    -----------
    None

    Returns
    -------
    None
    
    """
    model_name_method_path = os.path.join(models_path, name, method)
    figure_name_method_path = os.path.join(figures_path, name, method)
    results_list = os.listdir(results_path)

    for each_file in results_list:
        if fnmatch.fnmatch(each_file, name + "_" + method + ".csv"):
            file_name = each_file
    file_name = None
    sure = input(
        "You are about to delete all files in the following folder \n"
        + f"1. {model_name_method_path} \n"
        + f"2. {figure_name_method_path}\n"
        + "Additionally you will also delete \n"
        + f"3. {file_name} in the results folder \n"
        + "Type 'yes' or 'no' \n"
    )

    if sure == "yes":
        shutil.rmtree(model_name_method_path)
        shutil.rmtree(figure_name_method_path)
        os.remove(os.path.join(results_path, name + "_" + method + ".csv"))
        print("Results deleted.")
    else:
        print("Aborted.")


def check_path(path):
    """
    Checks if the path to saved models already exists.

    Parameters
    ----------
    path : string
        Path to the saved models.
    """

    if os.path.exists(path):
        if platform_release() != "eve":
            answer = (
                input(
                    "A model of this UID already exists. Do you wish to continue? y/n (enter: y): "
                )
                or "y"
            )
            if answer == "y":
                print("The existing model and associated results will be overwritten.")
            else:
                raise FileExistsError(
                    "Directory already exists. Please change your configuration in the run script. Aborting..."
                )
        else:
            print("Existing project folder will be overwritten... ")

# Soil moisture module for digital soil moisture mapping.

The module was developed in the study of Houben et al. (2025) published in Vadose Zone Journal:

Houben, T., Ebeling, P., Khurana, S.,  Schmid, J., Boog, J., (2025): *Machine-learning based spatio-temporal prediction of soil moisture in a grassland hillslope*. Vadose Zone Journal.

The input data and results of the study are published via Zenodo: [Digital Soil Moisture Mapping - Data accompanying Houben et al. 2025, Vadose Zone Journal](10.5281/zenodo.14871615)

## Folders in this repository

### SM_module
Contains the module with all functions and scripts to run the machine learning models.

### eval_scripts
Contains scripts to evaluate the machine learning models.

### run_scripts
Contains scripts to run the machine learning models which are presented in the manuscript.

```
.
├── AUTHORS.md 
├── LICENSE
├── README.md
└── SM_module
    ├── SM
    │   ├── cfg.py
    │   ├── eval.py
    │   ├── io.py
    │   ├── maps.py
    │   ├── misc.py
    │   ├── process.py
    │   ├── training.py
    │   └── vis.py
    ├── eval_scripts
    │   ├── I60_20210531_SM_example_Evaluation_PE.py
    │   ├── I69_20211127_RF_seed12000_JB_maps_vis.py
    │   ├── I85_20211124_GBRT_seed12000_PE_maps_vis.py
    │   └── I85_20211124_NN_seed12000_TH_maps_vis.py
    ├── ml-project-sm.yml
    ├── run_scripts_air_temp
    │   ├── I41_SVR_test_seed_manuscript.py
    │   ├── I69_20210614_Fit-SpatioTemp-RF_JB.py
    │   ├── I71_20211124_SM_GBRT_basefeat_tune_seeds_eve_PE.py
    │   ├── I74_20210630_SM_run_NN_54_12000_TH.py
    │   ├── I74_20210630_SM_run_NN_54_1337_TH.py
    │   ├── I74_20210630_SM_run_NN_54_420_TH.py
    │   ├── I74_20210630_SM_run_NN_54_42_TH.py
    │   └── I74_20210630_SM_run_NN_54_7_TH.py
    └── setup.py
```

## Installation
Create the ml-project-sm environment and install dependencies with the following command:

```
conda env create -f SM_module/ml-project-sm.yml
```
Activate the environment with the following command:

```
conda activate ml-project-sm
```

Then install the SM_module with pip (maybe install/upgrade pip before):

```
cd SM_module
pip install .
```

If you want to be able to adapt the package code and have the changes available in your environment (kernel reload required), use the following flag:

```
pip install -e .
```


## License Information

This product uses third-party dependencies with their respective licenses listed in the LICENSE file.
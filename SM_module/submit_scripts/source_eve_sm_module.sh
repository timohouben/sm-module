#!/bin/bash
#SBATCH -D /data/ml-cafe/project_soilmoisture/repo/ml-cafe_project_soilmoisture/SM_module
#SBATCH -J feat_select
#SBATCH -t 1000
#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH -o /data/ml-cafe/project_soilmoisture/submit_scripts/sk/feat_select.OUT
#SBATCH -e /data/ml-cafe/project_soilmoisture/submit_scripts/sk/feat_select.ERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=swamini.khurana@ufz.de

module load Anaconda3/5.3.0
module load foss/2018b

source activate /data/ml-cafe/project_soilmoisture/env/ml-project-sm

python ./run_scripts/SVR_feature_selection_eve-SK.py

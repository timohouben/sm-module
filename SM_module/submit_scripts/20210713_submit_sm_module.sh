## EXAMPLE SUBMIT SCRIPT FOR EVE AT UFZ
## REPLACE WITH YOUR CONFIGURATION

#!/bin/bash
#SBATCH -D /work/houben/nn-daily
#SBATCH -J I74_nn-daily
#SBATCH -t 3600
#SBATCH -n 1
#SBATCH --mem-per-cpu=8G
#SBATCH -o /work/houben/nn-daily/I74_test.OUT
#SBATCH -e /work/houben/nn-daily/I74_test.ERR

module load foss/2020b
module load Anaconda3

source /software/easybuild-broadwell/software/Core/Anaconda3/5.3.0/etc/profile.d/conda.sh

conda activate /data/ml-cafe/project_soilmoisture/env/ml-project-sm


echo "Running script with python:"
which python3

python3 /work/houben/ml-cafe_project_soilmoisture/SM_module/run_scripts/I74_20210630_SM_run_NN_TH.py
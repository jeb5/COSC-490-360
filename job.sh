#!/bin/bash
#SBATCH --account=nicje229
#SBATCH --job-name=COSC490_360_Experiments
#SBATCH --partition=aoraki_gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=job_%j.log

module load python/3.10.8
module load cuda/11.8

cd ..

python -m venv 360_env
source 360_env/bin/activate

cd -

/bin/bash setup.sh


python -u scripts/run_experiments.py

deactivate
echo "Job finished."
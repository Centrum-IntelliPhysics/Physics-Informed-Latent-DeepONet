#!/bin/bash -l

#SBATCH
#SBATCH --job-name=check
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -A sgoswam4_gpu
#SBATCH --mail-user=skarumu1@jh.edu
#SBATCH --mail-type=ALL


ml anaconda
conda activate py311


# Get parameters from the environment variables
seed=$seed
n_used=$n_used
save=$save

# Construct the output directory path
resultdir="results/a_Vanilla-NO/seed=${seed}_n_used=${n_used}"
mkdir -p $resultdir

# Run the papermill command with the passed parameters and save the output notebook in the constructed directory
papermill a_Vanilla-NO.ipynb "${resultdir}/output_seed=${seed}_n_used=${n_used}.ipynb" -p seed $seed -p n_used $n_used -p save $save > /dev/null 2>&1

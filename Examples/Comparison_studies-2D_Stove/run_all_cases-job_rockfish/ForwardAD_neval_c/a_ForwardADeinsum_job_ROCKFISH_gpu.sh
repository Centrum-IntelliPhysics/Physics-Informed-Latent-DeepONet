#!/bin/bash -l

#SBATCH
#SBATCH --job-name=check
#SBATCH --time=02-00:00:00
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -A sgoswam4_gpu
#SBATCH --mail-user=skarumu1@jh.edu
#SBATCH --mail-type=ALL


ml anaconda
conda activate py311


# Get parameters from the environment variables
neval_t=$neval_t
neval_x=$neval_x

# Construct the output directory path
resultdir="results/ForwardAD_neval_c/a_Vanilla-NO_ForwardADeinsum"
mkdir -p $resultdir

# Run the papermill command with the passed parameters and save the output notebook in the constructed directory
papermill a_Vanilla-NO_ForwardADeinsum.ipynb "${resultdir}/output_neval_t=${neval_t}_neval_x=${neval_x}.ipynb" -p neval_t $neval_t -p neval_x $neval_x > /dev/null 2>&1

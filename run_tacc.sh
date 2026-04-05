#!/bin/bash

#SBATCH -J training                          # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gh-dev                            # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                                 # Total number of mpi tasks requested
#SBATCH -t 2:00:00                           # Run time (hh:mm:ss)
#SBATCH -A AST24021							 # Project charge code

# Load CUDA module
module load cuda/12.8

# Load tacc-surrogates libraries
source /scratch/py-envs/tacc-surrogates/bin/activate
export PYTHONPATH=$SCRATCH/scripts/tacc-surrogates

# Run flow bench training/evaluation
python -W ignore epidemic_transfer_learning.py
#python -W ignore model_analysis.py
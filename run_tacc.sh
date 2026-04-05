#!/bin/bash

#SBATCH -J astrochem-surr-train              # Job name
#SBATCH -o log.%j                         	 # Name of stdout output file (%j expands to jobId)
#SBATCH -p gh-dev                            # Queue name
#SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
#SBATCH -n 1                                 # Total number of mpi tasks requested
#SBATCH -t 0:30:00                           # Run time (hh:mm:ss)
#SBATCH -A AST24021							 # Project charge code

module load gcc
module load python3_mpi/3.11.8
source /work/09338/carlos9/vista/AstroChemNet/.venv/bin/activate
python run.py train latent_ode --dataset uclchem_grav
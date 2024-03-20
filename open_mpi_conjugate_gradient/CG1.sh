#!/bin/bash -l
#SBATCH -M Rackham
#SBATCH -A SNIC2022-22-851
#SBATCH -t 5:00

module load gcc openmpi
mpirun ./CG1 1024

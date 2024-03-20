#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A uppmax2022-2-11
#SBATCH -t 10:00

module load gcc openmpi
mpirun ./matmul input18000.txt output.txt

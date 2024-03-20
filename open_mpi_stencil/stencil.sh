#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A uppmax2022-2-11
#SBATCH -t 5:00

module load gcc openmpi
mpirun ./stencil input16000000.txt output.txt 100

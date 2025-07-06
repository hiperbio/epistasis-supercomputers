#!/bin/bash
#SBATCH --job-name=gpuJob
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --partition=standard-g

module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm
module load cray-mpich

rocm-smi

make -f Makefile_mi250x

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/gsl/lib

srun ./bin/mpi-episdet_mi250x datasets/8192snps_32768samples.csv
# srun ./bin/mpi-episdet_mi250x datasets/16384snps_32768samples.csv

#!/bin/bash -l
#SBATCH --job-name=gpuJob
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4

module load OpenMPI/5.0.3-GCC-13.3.0
module load CUDA/12.6.0

nvidia-smi

make -f Makefile_a100

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/gsl/lib

srun ./bin/mpi-episdet_a100 datasets/8192snps_32768samples.csv
# srun ./bin/mpi-episdet_a100 datasets/16384snps_32768samples.csv


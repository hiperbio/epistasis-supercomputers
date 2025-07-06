#!/bin/bash -l
#SBATCH --job-name=gpuJob
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=17
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --gres=gpu:4

module load OpenMPI/5.0.3-GCC-13.3.0
module load CUDA/12.6.0

nvidia-smi

make -f Makefile_a100 triplets_k2

srun ./bin/mpi-tensor-episdet_a100.triplets.k2.bin datasets/db_16384snps_524288samples.txt
# srun ./bin/mpi-tensor-episdet_a100.triplets.k2.bin datasets/db_32768snps_524288samples.txt

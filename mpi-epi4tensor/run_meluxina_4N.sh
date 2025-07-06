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

make -f Makefile_a100 sm80_and

srun ./bin/mpi-epi4tensor_a100 datasets/db_2048snps_524288samples.txt
# srun ./bin/mpi-epi4tensor_a100 datasets/db_4096snps_524288samples.txt


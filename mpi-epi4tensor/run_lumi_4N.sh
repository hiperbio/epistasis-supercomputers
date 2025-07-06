#!/bin/bash
#SBATCH --job-name=gpuJob
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=33
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=map_gpu:0,1,2,3,4,5,6,7


module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.0.3
module load cray-mpich

rocm-smi 

make -f Makefile_mi250x

srun ./bin/mpi-epi4tensor_mi250x datasets/db_2048snps_32768samples.txt
# srun ./bin/mpi-epi4tensor_mi250x datasets/db_4096snps_32768samples.txt


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
module load rocm/6.0.3
module load cray-mpich

rocm-smi 

. /projappl/project_465001962/intel_oneapi_2025_01/setvars.sh --include-intel-llvm

sycl-ls

make -f Makefile_mi250x

export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/gsl/lib

srun ./bin/mpi-crossarch-episdet_mi250x datasets/8192snps_32768samples.csv
# srun ./bin/mpi-crossarch-episdet_mi250x datasets/16384snps_32768samples.csv


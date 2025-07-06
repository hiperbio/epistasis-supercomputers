# epistasis-supercomputers

This repository includes MPI-extended versions of GPU-accelerated epistasis detection codes, adding support for multi-device execution on multiple nodes.
These codes have been developed in the context of advanced computing projects granting access to the GPU partitions of the MeluXina and LUMI EuroHPC supercomputers.
Codes that were originally developed in CUDA have been ported to HIP as a means to enable targeting the AMD MI250X GPUs available on LUMI.

## Setup

The GNU Scientific Library is used in `mpi-episdet` and in `mpi-crossarch-episdet`, and the CUTLASS library is used in the CUDA implementations of `mpi-tensor-episdet` and `mpi-epi4tensor`.
The dependencies required for executing each tool can be installed with the corresponding `setup.sh` script.

## Usage

The MPI-extended codes are accompanied by scripts for launching jobs that process synthetic datasets on MeluXina or LUMI.
These scripts can be easily adapted to process different datasets and/or target different amounts of supercomputer nodes.
Using other supercomputers might require loading other Linux environment modules and/or adapting the corresponding makefiles.

The `mpi-episdet` and `mpi-crossarch-episdet` tools receive as input datasets represented in CSV format, while `mpi-tensor-episdet` and `mpi-epi4tensor` process datasets represented in a binarized format. Download datasets referenced in the scripts accompanying the epistasis detection tools from <a href="https://drive.google.com/file/d/1s5VpZdubNMzPULj4xN4vDup6xWTclILV/view?usp=sharing">here</a>.


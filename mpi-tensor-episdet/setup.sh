#!/bin/bash
currentdir=$PWD
wget --no-check-certificate https://github.com/NVIDIA/cutlass/archive/refs/tags/v1.3.3.tar.gz
tar xvf v1.3.3.tar.gz
rm -f v1.3.3.tar.gz


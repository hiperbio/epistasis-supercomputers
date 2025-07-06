#!/bin/bash
currentdir=$PWD
wget --no-check-certificate https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.4.1.tar.gz
tar xvf v3.4.1.tar.gz
rm -f v3.4.1.tar.gz


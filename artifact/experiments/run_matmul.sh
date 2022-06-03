#!/bin/bash

echo "Building cuda binaries ..."
nvcc -O3 --std c++17 -expt-relaxed-constexpr -arch native experiments/layout/device/main.cu -I../noarr-structures/include -I../noarr-pipelines/include -Iexperiments/layout/device -I. -o ecoopex-gpu
echo "done"

echo "Running CUDA matrix multiplication (custom matrix) ..."
./ecoopex-gpu $1 $2 $3 2> matmul_custom.csv

echo "GPU tests are complete"

if which Rscript > /dev/null 2> /dev/null; then
	echo "Plotting graphs with R ..."
	mkdir -p ./plots
	Rscript ./plot_matmul.R $1 $2 $3
else
	echo "Rscript not found, skipping graph plotting!"
fi

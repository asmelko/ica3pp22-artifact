#!/bin/bash

echo "Building c++ binaries ..."
g++ -O3 -DNDEBUG examples/main.cpp examples/stencil.cpp -I../noarr-structures/include -I. -o runex
echo "done"

echo "Building cuda binaries ..."
nvcc -O3 --std c++17 -expt-relaxed-constexpr -arch native experiments/layout/device/main.cu -I../noarr-structures/include -I../noarr-pipelines/include -Iexperiments/layout/device -I. -o runex-gpu
echo "done"

echo "Running CPU stencil test ..."
./runex $1 > stencil.csv

echo "Running CPU stencil test (constexpr version) ..."
./runex cloop $1 > stencil_loop.csv

echo "CPU tests are complete"

echo "Running CUDA matrix multiplication (small matrix) ..."
./runex-gpu 1008 1008 1008 2> matmul_small.csv

if [[ "$1" != "--quick" ]]; then
	echo "Running CUDA matrix multiplication (large matrix) ..."
	echo "(this might take up to a few hours)"
	./runex-gpu 10080 10080 10080 2> matmul_big.csv

	echo "GPU tests are complete"

	if which Rscript > /dev/null 2> /dev/null; then
		echo "Plotting graphs with R ..."
		mkdir -p ./plots
		Rscript ./plots.R ENABLE_CUDA
	else
		echo "Rscript not found, skipping graph plotting!"
	fi
fi

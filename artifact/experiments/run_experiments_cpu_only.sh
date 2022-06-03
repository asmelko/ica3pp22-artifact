#!/bin/bash

echo "Building c++ binaries ..."
g++ -O3 -DNDEBUG examples/main.cpp examples/stencil.cpp -I../noarr-structures/include -I. -o runex
echo "done"

echo "Running CPU stencil test ..."
./runex $1 > stencil.csv

echo "Running CPU stencil test (constexpr version) ..."
./runex cloop $1 > stencil_loop.csv

if [[ "$1" != "--quick" ]]; then
	if which Rscript > /dev/null 2> /dev/null; then
		echo "Plotting graphs with R ..."
		mkdir -p ./plots
		Rscript ./plots.R
	else
		echo "Rscript not found, skipping graph plotting!"
	fi
fi

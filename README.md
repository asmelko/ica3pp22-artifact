# Artifact Submission: noarr-structures

This is a repliaction package containing code and experimental results related to a ICA3PP paper titled:  **Astute Approach to Handling Memory Layouts of Regular Data Structures**
<!-- ICA3PP submission number for the paper: **75** -->

## Overview

The artifact comprises the following directories:

* `noarr-structures` -- C++ header-only library which is described in the paper.
* `noarr-pipelines` -- Additional helper library that wraps GPU management (memory allocation, kernel execution ...).
* `experiments` -- Experimental code and scripts to compile it and executes the measurements of various aspects of noarr-structures (presented in paper).
* `data-plots` -- The directory containing plots either directly present in the paper or just mentioned due to the page limit. The directory also contains csv measurements files that generated the plots.

The `noarr` library also contains readme files with examples and documentation

* `noarr-structures/README.md` -- main overview and quick start guidelines
* `noarr-structures/noarr_docs_user/README.md` -- users' documentation (i.e., how to use the library)
* `noarr-structures/noarr_docs_tech/README.md` -- technical documentation (i.e., how the library works inside)
* `noarr-structures/examples/matrix` -- a complete example (and readme) of using noarr for matrix management

## Detailed artifact contents

The artifact contains scripts that re-create all measurements published in the paper. Please note that all experiments are performance-based, so actual measured values are heavily influenced by your hardware configuration. Nevertheless, for common configurations it should yield results which would support the conclusions we reached in the paper.

The aforementioned figures can be generated simply by executing `run_experiments_all.sh` in `experiments` directory. The figures will appear in `plots` sub-directory:

* `Fig1-matmul.pdf` -- quick overview of selected layouts and how they affect performance of matrix multiplication
* `Fig2-matmul_transform.pdf` -- depicts overhead of matrix layout transformation when applied on matrix multiplication
* `Fig3-stencil.pdf` -- comparison of two versions of 3D stencil (100 iterations); dynamic = indices are computed in each iteration, static = noarr allows some const. propagation, so some indices may be precomputed (even at compile time)
* `Fig4-stencil_loop.pdf` -- extension of the stencil experiment (indexing = same as static in Fig5, indexing+looping = experiment modified so the loops have constant boundaries that allow additional optimizations)
* `FigX-heatmap_all.pdf` (not included in the paper) -- compares all possible layouts of (two) input matrices and output matrix in matrix multiplication

For convenience, these figures were pregenerated and can be found in `data-plots` directory together with data that generated them.

The artifact also presents the whole `noarr` library which is a prototype of our proposed approach to memory layout management of data structures in C++. It is associated with documentation readme files that should help the reader to start using this library quickly. Furthermore, there is an example `noarr-structures/examples/matrix` which can be used as a starting point to test the capabilities of the library. The most current version is kept in [repositories](https://github.com/ParaCoToUl) on GitHub under MIT license.

## Requirements for running the experiments

Hardware requirements:

* To replicate all tests, a CUDA-compatible GPU is required.

Software requirements:

* `g++` (version 11.x)
* [CUDA toolkit 11.6 or later](https://developer.nvidia.com/cuda-downloads) and appropriate driver (for GPU tests)
* `R` software for plotting the graphs (see details below)

Installing R on RHEL/CentOS-like systems:
```
sudo dnf install R
```

Installing R on Debian/Ubuntu:
```
sudo apt-get update && apt-get install -y r-base
```

Afterwards, R packages needs to be installed:
```
sudo R -e "install.packages('ggplot2', repos='https://cloud.r-project.org')"
sudo R -e "install.packages('cowplot', repos='https://cloud.r-project.org')"
sudo R -e "install.packages('sitools', repos='https://cloud.r-project.org')" 
```

## Experiments execution

**Kick the tires:**

Just to see whether the code is working, run the script with `--quick` argument from `experiments` directory:
```
chmod +x run_experiments_all.sh
./run_experiments_all.sh --quick
```
or without GPU:
```
chmod +x run_experiments_cpu_only.sh
./run_experiments_cpu_only.sh --quick
```
This will not produce any graphs, but it is a quick verification whether code is running. The script generates 3 csv files (or 2, if `cpu_only` version is run). They are `stencil.csv` and `stencil_loop.csv` for CPU measurements (200 entries), and `matmul_small.csv` for GPU measurements (4320 entries). Further documentation of the measurement files structure is present in `data-plots/data/README.md`.


**Full measurement:**

To execute all experiments and create all the plots:
```
./run_experiments_all.sh
```
Please note that this could take up to several hours depending on your hardware (e.g., nearly 2 hours on our ref. hardware with NVIDIA V100 GPU).

Without a CUDA-compatible GPU, you can run:
```
./run_experiments_cpu_only.sh
```
which will measure data (and recreate plots) only for Figure 3 and 4. This might take about 10 minutes. 

When the script finishes, figures will appear in `experiments/plots` directory.


**Custom matrix multiplication measurements:**

The artifact also contains the script that runs GPU matrix multiplication kernel with specified sizes of input matrices and generates `heatmap_custom.pdf` (a heatmap of all possible layouts). The script expects exactly three parameters and can be run as such (only sizes that are multiples of 16 are supported):
```
./run_matmul.sh <LEFT_INPUT_MATRIX_HEIGHT> <LEFT_INPUT_MATRIX_WIDTH> <RIGHT_INPUT_MATRIX_WIDTH>
```

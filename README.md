This repository currently contains an early working draft of the LJR XPCI Python implementation. The code was saved to preserve the development state and is being reorganised into a cleaner version. Interfaces, file structure, and example scripts may change.




# LJR XPCI

Python implementation of the Linear Joint Reconstruction (LJR) pipeline for single-shot edge-illumination X-ray phase-contrast computed tomography.

## Installation

Create and activate the conda environment:

conda create -n ljr_xpci python=3.11

conda activate ljr_xpci

Install the required packages:

conda install -c conda-forge numpy scipy matplotlib tifffile imagecodecs

conda install -c astra-toolbox -c nvidia astra-toolbox

Install the package from the repository root:

pip install -e .

## Required packages

- numpy
- scipy
- matplotlib
- tifffile
- imagecodecs
- astra-toolbox

## Running an example dataset

The repository includes a project script for running the LJR pipeline on an example dataset.

Run from the repository root:

python projects/optica_2026_ljr/scripts/run_ljr_khush.py --data-root /path/to/remotedata_folder

The data root is the folder that contains the dataset folders used by the project. For the current test case, it must contain:

pd/Khush_folder/DATA.tif

The name Khush refers only to this specific test dataset. The LJR functions in src/ljr_xpci are general and can be reused with other datasets.


## Local paths

Machine-specific paths should not be hardcoded inside src/ljr_xpci.

The src/ljr_xpci folder contains reusable package code. Private local paths should stay inside:

projects/optica_2026_ljr/configs/local_paths.py

This file can be adapted locally by each user.

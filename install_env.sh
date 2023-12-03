#!/bin/bash
# 
# Installer for refrtomo
# 
# Run: ./install.sh
# 
# M. Ravasi, 03/12/2023

echo 'Creating refrtomo environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate refrtomo
conda env list
echo 'Created and activated environment:' $(which python)

# check pylops work as expected
echo 'Checking pylops version and running a command...'
python -c 'import pylops as pl; print(pl.__version__); print(pl.Zero(5))'

echo 'Done!'


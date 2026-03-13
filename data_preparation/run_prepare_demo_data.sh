#!/bin/bash
#SBATCH --job-name=prepare_demo_data
#SBATCH --partition=CPUQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/home/haroldh/spGDMM/bayesian-hierarchical-beta-diversity-modelling/prepare_demo_data_%j.log

source ~/.bashrc
conda activate spgdmm-demo

cd /cluster/home/haroldh/spGDMM/bayesian-hierarchical-beta-diversity-modelling
python prepare_demo_data.py

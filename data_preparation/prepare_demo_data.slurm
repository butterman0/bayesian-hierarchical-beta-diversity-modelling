#!/bin/bash
#SBATCH --job-name=prepare_demo
#SBATCH --output=/cluster/home/haroldh/spGDMM/bayesian-hierarchical-beta-diversity-modelling/prepare_demo_%j.log
#SBATCH --error=/cluster/home/haroldh/spGDMM/bayesian-hierarchical-beta-diversity-modelling/prepare_demo_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

source /cluster/home/haroldh/miniforge3/etc/profile.d/conda.sh
conda activate spgdmm-demo

cd /cluster/home/haroldh/spGDMM/bayesian-hierarchical-beta-diversity-modelling
python -u prepare_demo_data.py

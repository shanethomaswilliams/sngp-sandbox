#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu
#SBATCH --job-name=sngp_grid_search
#SBATCH --output=/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/sngp_grid_search_%j.out
#SBATCH --error=/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/sngp_grid_search_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G

module load cuda/11.0
source activate DeepLP

python /cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/notebooks/train_models.py
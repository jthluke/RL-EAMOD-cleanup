#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=2880
#SBATCH --partition=gpu 
#SBATCH --gpus=1 

conda activate
python main.py --test True --max_episodes 32000
python main.py --test True --max_episodes 32000 --equal_distr_baseline True

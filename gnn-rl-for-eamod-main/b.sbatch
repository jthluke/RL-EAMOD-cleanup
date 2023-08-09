#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate
python main.py --max_episodes 20000
python main.py --test True --max_episodes 20000

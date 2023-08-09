#!/usr/bin/bash
#SBATCH --job-name=pure_heuristic_equal_distr
#SBATCH --output=pure_heuristic_equal_distr.%j.out
#SBATCH --error=pure_heuristic_equal_distr.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate
python main.py --max_episodes 20000 --equal_distr_baseline True
python main.py --test True --max_episodes 20000 --equal_distr_baseline True

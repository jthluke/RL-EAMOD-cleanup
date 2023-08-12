#!/usr/bin/bash
#SBATCH --job-name=pure_rl
#SBATCH --output=pure_rl.%j.out
#SBATCH --error=pure_rl.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 30

conda activate
python main.py --max_episodes 20000
python main.py --test True --max_episodes 1000

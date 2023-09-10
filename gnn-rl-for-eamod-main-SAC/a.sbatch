#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 30

conda activate
python main.py --spatial_nodes 5 --T 24
python main.py --spatial_nodes 10 --T 24
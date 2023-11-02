#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=4-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 3

conda activate
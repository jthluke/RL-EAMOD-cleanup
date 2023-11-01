#!/usr/bin/bash
#SBATCH --job-name=rl_10
#SBATCH --output=rl_10.%j.out
#SBATCH --error=rl_10.%j.err
#SBATCH --time=4-00:00:00
#SBATCH --qos=long
#SBATCH -c 32
#SBATCH -G 1
#SBATCH --mem=30GB
#SBATCH -C GPU_MEM:40GB

conda activate

python main.py --spatial_nodes 10 --T 48 --city NY --resume True --gurobi Daniele2
python main.py --spatial_nodes 10 --T 48 --city SF --resume True --gurobi Daniele2


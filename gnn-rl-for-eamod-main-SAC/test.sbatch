#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=4-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 3

conda activate

python main.py --spatial_nodes 5 --T 48 --city NY --zeroShotCity True --gurobi Daniele3
python main.py --spatial_nodes 5 --T 48 --city SF --zeroShotCity True --gurobi Daniele3

python main.py --spatial_nodes 10 --T 48 --city NY --zeroShotNodes True --gurobi Daniele3
python main.py --spatial_nodes 15 --T 48 --city NY --zeroShotNodes True --gurobi Daniele3

python main.py --spatial_nodes 10 --T 48 --city SF --zeroShotNodes True --gurobi Daniele3
python main.py --spatial_nodes 15 --T 48 --city SF --zeroShotNodes True --gurobi Daniele3
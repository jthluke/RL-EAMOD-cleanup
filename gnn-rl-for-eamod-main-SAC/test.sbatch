#!/usr/bin/bash
#SBATCH --job-name=rl_test
#SBATCH --output=rl_test.%j.out
#SBATCH --error=rl_test.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH -c 5
#SBATCH -G 1
#SBATCH --mem=30GB

conda activate

python main.py --spatial_nodes 5 --T 48 --city NY --zeroShotCity True --gurobi Daniele3
python main.py --spatial_nodes 5 --T 48 --city SF --zeroShotCity True --gurobi Daniele3

python main.py --spatial_nodes 10 --T 48 --city NY --zeroShotNodes True --gurobi Daniele3
python main.py --spatial_nodes 15 --T 48 --city NY --zeroShotNodes True --gurobi Daniele3

python main.py --spatial_nodes 10 --T 48 --city SF --zeroShotNodes True --gurobi Daniele3
python main.py --spatial_nodes 15 --T 48 --city SF --zeroShotNodes True --gurobi Daniele3
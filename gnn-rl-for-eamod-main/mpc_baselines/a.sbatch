#!/usr/bin/bash
#SBATCH --job-name=mpc
#SBATCH --output=mpc.%j.out
#SBATCH --error=mpc.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH -c 32

python main_mpc.py --spatial_nodes 5 --mpc_horizon 48 --city NY --gurobi Aaryan
python main_mpc.py --spatial_nodes 5 --mpc_horizon 2 --city NY --noisy True --gurobi Aaryan
python main_mpc.py --spatial_nodes 5 --mpc_horizon 1 --city NY --noisy True --gurobi Aaryan
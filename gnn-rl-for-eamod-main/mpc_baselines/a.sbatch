#!/usr/bin/bash
#SBATCH --job-name=mpc
#SBATCH --output=mpc.%j.out
#SBATCH --error=mpc.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 30

conda activate

python main_mpc.py --spatial_nodes 5 --mpc_horizon 24
python main_mpc.py --spatial_nodes 10 --mpc_horizon 24
python main_mpc.py --spatial_nodes 15 --mpc_horizon 24
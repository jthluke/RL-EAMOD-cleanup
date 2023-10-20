#!/usr/bin/bash
#SBATCH --job-name=mpc
#SBATCH --output=mpc.%j.out
#SBATCH --error=mpc.%j.err
#SBATCH --time=3-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 32

conda activate

python main_mpc.py --spatial_nodes 5 --mpc_horizon 48 --city NY
python main_mpc.py --spatial_nodes 5 --mpc_horizon 48 --city NY --noisy True

python main_mpc.py --spatial_nodes 10 --mpc_horizon 36 --city NY
python main_mpc.py --spatial_nodes 10 --mpc_horizon 36 --city NY --noisy True
python main_mpc.py --spatial_nodes 10 --mpc_horizon 24 --city NY
python main_mpc.py --spatial_nodes 10 --mpc_horizon 24 --city NY --noisy True
python main_mpc.py --spatial_nodes 10 --mpc_horizon 12 --city NY
python main_mpc.py --spatial_nodes 10 --mpc_horizon 12 --city NY --noisy True

python main_mpc.py --spatial_nodes 15 --mpc_horizon 24 --city NY
python main_mpc.py --spatial_nodes 15 --mpc_horizon 24 --city NY --noisy True
python main_mpc.py --spatial_nodes 15 --mpc_horizon 12 --city NY
python main_mpc.py --spatial_nodes 15 --mpc_horizon 12 --city NY --noisy True
python main_mpc.py --spatial_nodes 15 --mpc_horizon 6 --city NY
python main_mpc.py --spatial_nodes 15 --mpc_horizon 6 --city NY --noisy True

python main_mpc.py --spatial_nodes 20 --mpc_horizon 12 --city NY
python main_mpc.py --spatial_nodes 20 --mpc_horizon 12 --city NY --noisy True
python main_mpc.py --spatial_nodes 20 --mpc_horizon 6 --city NY
python main_mpc.py --spatial_nodes 20 --mpc_horizon 6 --city NY --noisy True
python main_mpc.py --spatial_nodes 20 --mpc_horizon 3 --city NY
python main_mpc.py --spatial_nodes 20 --mpc_horizon 3 --city NY --noisy True




python main_mpc.py --spatial_nodes 5 --mpc_horizon 48 --city SF
python main_mpc.py --spatial_nodes 5 --mpc_horizon 48 --city SF --noisy True

python main_mpc.py --spatial_nodes 10 --mpc_horizon 36 --city SF
python main_mpc.py --spatial_nodes 10 --mpc_horizon 36 --city SF --noisy True
python main_mpc.py --spatial_nodes 10 --mpc_horizon 24 --city SF
python main_mpc.py --spatial_nodes 10 --mpc_horizon 24 --city SF --noisy True
python main_mpc.py --spatial_nodes 10 --mpc_horizon 12 --city SF
python main_mpc.py --spatial_nodes 10 --mpc_horizon 12 --city SF --noisy True

python main_mpc.py --spatial_nodes 15 --mpc_horizon 24 --city SF
python main_mpc.py --spatial_nodes 15 --mpc_horizon 24 --city SF --noisy True
python main_mpc.py --spatial_nodes 15 --mpc_horizon 12 --city SF
python main_mpc.py --spatial_nodes 15 --mpc_horizon 12 --city SF --noisy True
python main_mpc.py --spatial_nodes 15 --mpc_horizon 6 --city SF
python main_mpc.py --spatial_nodes 15 --mpc_horizon 6 --city SF --noisy True

python main_mpc.py --spatial_nodes 20 --mpc_horizon 12 --city SF
python main_mpc.py --spatial_nodes 20 --mpc_horizon 12 --city SF --noisy True
python main_mpc.py --spatial_nodes 20 --mpc_horizon 6 --city SF
python main_mpc.py --spatial_nodes 20 --mpc_horizon 6 --city SF --noisy True
python main_mpc.py --spatial_nodes 20 --mpc_horizon 3 --city SF
python main_mpc.py --spatial_nodes 20 --mpc_horizon 3 --city SF --noisy True
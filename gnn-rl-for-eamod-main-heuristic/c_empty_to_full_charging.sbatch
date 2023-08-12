#!/usr/bin/bash
#SBATCH --job-name=charge_heuristic_spatial_rl_empty_to_full
#SBATCH --output=charge_heuristic_spatial_rl_empty_to_full.%j.out
#SBATCH --error=charge_heuristic_spatial_rl_empty_to_full.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate

python main.py --charging_heuristic empty_to_full --max_episodes 20000

# tags = charge_heuristic, spatial_rl, c_batch, spatial_graph
# job not submitted
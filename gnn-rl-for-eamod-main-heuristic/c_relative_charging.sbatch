#!/usr/bin/bash
#SBATCH --job-name=charge_heuristic_spatial_rl_relative_charging
#SBATCH --output=charge_heuristic_spatial_rl_relative_charging.%j.out
#SBATCH --error=charge_heuristic_spatial_rl_relative_charging.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate

python main.py --charging_heuristic relative_charging --max_episodes 20000

# tags = charge_heuristic, spatial_rl, c_batch, spatial_graph
# job not submitted
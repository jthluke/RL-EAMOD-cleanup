#!/usr/bin/bash
#SBATCH --job-name=charge_heuristic_spatial_rl
#SBATCH --output=charge_heuristic_spatial_rl.%j.out
#SBATCH --error=charge_heuristic_spatial_rl.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 7

conda activate

python main.py --charging_heuristic empty_to_full --max_episodes 20000
python main.py --charging_heuristic off_peak_charging --max_episodes 20000
python main.py --charging_heuristic off_peak_one_step --max_episodes 20000
python main.py --charging_heuristic relative_charging --max_episodes 20000

# tags = charge_heuristic, spatial_rl, c_batch, spatial_graph
# job submitted
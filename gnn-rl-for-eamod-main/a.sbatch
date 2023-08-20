#!/usr/bin/bash
#SBATCH --job-name=no_rl
#SBATCH --output=no_rl.%j.out
#SBATCH --error=no_rl.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 7

conda activate
python main.py --test True --max_episodes 10 --equal_distr_baseline True

# tags = charge_heuristic, spatial_heuristic, a_batch, test, space_charge_graph
# job complete = 
#!/usr/bin/bash
#SBATCH --job-name=partial_rl
#SBATCH --output=partial_rl.%j.out
#SBATCH --error=partial_rl.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate

python main.py --charging_heuristic empty_to_full --max_episodes 20000
python main.py --charging_heuristic off_peak_charging --max_episodes 20000
python main.py --charging_heuristic off_peak_charging --max_episodes 20000
python main.py --charging_heuristic off_peak_one_step --max_episodes 20000
python main.py --charging_heuristic relative_charging --max_episodes 20000
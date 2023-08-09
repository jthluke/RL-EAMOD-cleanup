#!/usr/bin/bash
#SBATCH --job-name=heuristic
#SBATCH --output=heuristic.%j.out
#SBATCH --error=heuristic.%j.err
#SBATCH --time=2800
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate
python main.py
# python main.py --equal_distr_baseline True
# python main.py --charging_heuristic off_peak_charging
# python main.py --charging_heuristic off_peak_charging --equal_distr_baseline True
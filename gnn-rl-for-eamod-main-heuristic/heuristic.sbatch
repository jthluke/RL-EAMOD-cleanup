#!/usr/bin/bash
#SBATCH --job-name=heuristic
#SBATCH --output=heuristic.%j.out
#SBATCH --error=heuristic.%j.err
#SBATCH --time=2880
#SBATCH --partition=gpu 
#SBATCH --gpus=1 

conda activate
python main.py --test True
python main.py --test True --equal_distr_baseline True
python main.py --test True --charging_heuristic off_peak_charging
python main.py --test True --charging_heuristic off_peak_charging --equal_distr_baseline True
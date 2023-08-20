#!/usr/bin/bash
#SBATCH --job-name=pure_heuristic
#SBATCH --output=pure_heuristic.%j.out
#SBATCH --error=pure_heuristic.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 30

conda activate

python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic off_peak_charging --equal_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --test True --max_episodes 10

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic off_peak_charging --prop_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --test True --max_episodes 10
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --test True --max_episodes 10

# tags = charge_heuristic, spatial_heuristic, a_batch, test, spatial_graph
# job complete
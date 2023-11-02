#!/usr/bin/bash
#SBATCH --job-name=he
#SBATCH --output=he.%j.out
#SBATCH --error=he.%j.err
#SBATCH --time=4-00:00:00
#SBATCH--qos long
#SBATCH --partition=normal
#SBATCH -c 2

conda activate

python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 5 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 5 --T 48 --city NY --max_episodes 5 --gurobi Aaryan


python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 10 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 10 --T 48 --city NY --max_episodes 5 --gurobi Aaryan


python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 15 --T 48 --city SF --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --equal_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan

python main.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_nothing --prop_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
python main.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 15 --T 48 --city NY --max_episodes 5 --gurobi Aaryan
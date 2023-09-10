#!/usr/bin/bash
#SBATCH --job-name=pure_heuristic
#SBATCH --output=pure_heuristic.%j.out
#SBATCH --error=pure_heuristic.%j.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 11

conda activate

python main_sf.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic off_peak_charging --equal_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 5 --T 24

python main_sf.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic off_peak_charging --prop_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 5 --T 24
python main_sf.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 5 --T 24

python main_sf.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic off_peak_charging --equal_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 10 --T 24

python main_sf.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic off_peak_charging --prop_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 10 --T 24
python main_sf.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 10 --T 24

python main_sf.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic off_peak_charging --equal_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 15 --T 24

python main_sf.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic off_peak_charging --prop_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 15 --T 24
python main_sf.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 15 --T 24

python main_sf.py --charging_heuristic empty_to_full --equal_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic off_peak_charging --equal_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --equal_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic relative_charging --equal_distr_baseline True --spatial_nodes 20 --T 24

python main_sf.py --charging_heuristic empty_to_full --prop_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic off_peak_charging --prop_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic off_peak_one_step --prop_distr_baseline True --spatial_nodes 20 --T 24
python main_sf.py --charging_heuristic relative_charging --prop_distr_baseline True --spatial_nodes 20 --T 24

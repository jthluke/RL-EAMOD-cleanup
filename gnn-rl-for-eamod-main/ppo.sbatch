#!/usr/bin/bash
#SBATCH --job-name=pure_rl_sac
#SBATCH --output=pure_rl_sac.%j.out
#SBATCH --error=pure_rl_sac.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 25

conda activate
python main_stable_baslines.py --max_episodes 20000
python main_stable_baslines.py --test True --max_episodes 100

# tags = charge_rl, spatial_rl, test_batch, (2nd) test, space_charge_graph
# job submitted

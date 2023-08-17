#!/usr/bin/bash
#SBATCH --job-name=pure_rl_mpnn
#SBATCH --output=pure_rl_mpnn.%j.out
#SBATCH --error=pure_rl_mpnn.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 25

conda activate
python main.py --max_episodes 20000
python main.py --test True --max_episodes 1000

# tags = charge_rl, spatial_rl, b_batch, (2nd) test, space_charge_graph
# job submitted

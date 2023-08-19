#!/usr/bin/bash
#SBATCH --job-name=pure_rl
#SBATCH --output=pure_rl.%j.out
#SBATCH --error=pure_rl.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 7

conda activate
python main.py --max_episodes 20000
python main.py --test True --max_episodes 100

# tags = charge_rl, spatial_rl, b_batch, (2nd) test, space_charge_graph
# job submitted

# iteration one complete = separation of rebreward internal added, also added charge penalty + wasted customer penalty
# https://wandb.ai/aaryan04/e-amod/runs/9ph8y8px/overview?workspace=user-aaryan04, https://wandb.ai/aaryan04/e-amod/runs/vl8dakr8?workspace=user-aaryan04
# these iterations outperform a.sbatch


# iteration two = added overconcentration penalty 

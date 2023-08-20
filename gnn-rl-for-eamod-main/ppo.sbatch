#!/usr/bin/bash
#SBATCH --job-name=pure_rl_sac
#SBATCH --output=pure_rl_sac.%j.out
#SBATCH --error=pure_rl_sac.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 7

conda activate
python main_stable_baselines.py --max_episodes 20000
python main_stable_baselines.py --test True --max_episodes 100

# tags = charge_rl, spatial_rl, test_batch, (2nd) test, space_charge_graph
# job submitted

# synced wih b.batch
# iteration 1 = https://wandb.ai/aaryan04/e-amod/runs/5ftydl4w/overview?workspace=user-aaryan04

# iteration 2 = 

# iteration 3 =

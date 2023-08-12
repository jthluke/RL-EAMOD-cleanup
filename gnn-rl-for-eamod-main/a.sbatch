#!/usr/bin/bash
#SBATCH --job-name=pure_heuristic_equal_distr
#SBATCH --output=pure_heuristic_equal_distr.%j.out
#SBATCH --error=pure_heuristic_equal_distr.%j.err
#SBATCH --time=300
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH -c 30

conda activate
python main.py --max_episodes 100 --equal_distr_baseline True
python main.py --test True --max_episodes 100 --equal_distr_baseline True

# tags = charge_heuristic, spatial_heuristic, a_batch, test, space_charge_graph
# job complete = https://wandb.ai/aaryan04/e-amod/runs/2w0b1xzx/overview?workspace=user-aaryan04, https://wandb.ai/aaryan04/e-amod/runs/xogktaax/overview?workspace=user-aaryan04

#!/home/groups/pavone/RL-EAMOD/RL-EAMOD-Public/gnn-rl-for-eamod-main-heuristic
#SBATCH --job-name=heuristic
#SBATCH --output=heuristic.%j.out
#SBATCH --error=heuristic.%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu 
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=25GB 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu_cmode=exclusive


python main.py --test True
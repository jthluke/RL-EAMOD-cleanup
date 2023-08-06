#!/home/groups/pavone/RL-EAMOD/RL-EAMOD-Public/gnn-rl-for-eamod-main-heuristic
#SBATCH --job-name=heuristic
#SBATCH --output=heuristic.%j.out
#SBATCH --error=heuristic.%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=gpu 
#SBATCH --gpus=4 
#SBATCH --mem-per-gpu=25GB 
#SBATCH --nodes=4 
#SBATCH --ntasks=4 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu_cmode=exclusive

conda activate
python main.py --test True
python main.py --test True --equal_distr_baseline True
python main.py --test True --charging_heuristic off_peak_charging
python main.py --test True --charging_heuristic off_peak_charging --equal_distr_baseline True
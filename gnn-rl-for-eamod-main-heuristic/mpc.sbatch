#SBATCH --job-name=heuristic
#SBATCH --output=heuristic.%j.out
#SBATCH --error=heuristic.%j.err
#SBATCH --time=30:00:00
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --mem=16GB


python main.py --test True
python main.py --test True --equal_distr_baseline True
python main.py --test True --charging_heuristic off_peak_charging
python main.py --test True --charging_heuristic off_peak_charging --equal_distr_baseline True
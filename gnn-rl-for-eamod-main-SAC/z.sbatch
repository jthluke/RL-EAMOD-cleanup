#!/usr/bin/bash
#SBATCH --job-name=rl
#SBATCH --output=rl.%j.out
#SBATCH --error=rl.%j.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --partition=normal
#SBATCH -c 11

conda activate

# zero shot nodes
python main.py --spatial_nodes 10 --T 48 --city SF --zeroShotNodes True --max_episodes 10
python main.py --spatial_nodes 10 --T 48 --city NY --zeroShotNodes True --max_episodes 10

python main.py --spatial_nodes 15 --T 48 --city SF --zeroShotNodes True --max_episodes 10
python main.py --spatial_nodes 15 --T 48 --city NY --zeroShotNodes True --max_episodes 10

# zero shot city
python main.py --spatial_nodes 5 --T 48 --city SF --zeroShotCity True --max_episodes 10
python main.py --spatial_nodes 5 --T 48 --city NY --zeroShotCity True --max_episodes 10

python main.py --spatial_nodes 10 --T 48 --city SF --zeroShotCity True --max_episodes 10
python main.py --spatial_nodes 10 --T 48 --city NY --zeroShotCity True --max_episodes 10

python main.py --spatial_nodes 15 --T 48 --city SF --zeroShotCity True --max_episodes 10
python main.py --spatial_nodes 15 --T 48 --city NY --zeroShotCity True --max_episodes 10

# python main.py --spatial_nodes 5 --T 48 --city NY
# python main.py --spatial_nodes 10 --T 48 --city NY
# python main.py --spatial_nodes 15 --T 48 --city NY
# python main.py --spatial_nodes 20 --T 48 --city NY
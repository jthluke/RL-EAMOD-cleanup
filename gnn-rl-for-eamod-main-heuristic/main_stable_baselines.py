import argparse
import gurobipy as gp
import gym
import os
import numpy as np
from src.algos.gym import FleetEnv
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO, SAC
import json

from src.envs.amod_env import Scenario, AMoD


parser = argparse.ArgumentParser(description='A2C-GNN')
# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--number_subproblems', type=int, default=3, metavar='N',
                    help='number of training subproblems (default: 3')
parser.add_argument('--T', type=int, default=10, metavar='N',
                    help='Time horizon for the A2C')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# TODO finish implementation
def create_scenario(json_file_path, energy_file_path):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    spatial_nodes = data['spacialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_time_per_level = data['chargeTime']
    chargers = []
    for node in range(spatial_nodes):
        chargers.append(node)

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, number_charge_levels=number_charge_levels, charge_time=charge_time_per_level, 
                        energy_distance=energy_dist, tf=tf, sd=args.seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc)
    return scenario


def main():
    # set Gurobi environment
    gurobi_env = gp.Env(empty=True)
    gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
    gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
    gurobi_env.setParam('LICENSEID', 799876)
    gurobi_env.setParam("OutputFlag",0)
    gurobi_env.start()

    problem_folder = 'SF_10_clustered'
    if args.toy:
        problem_folder = 'Toy'
        experiment = 'stable_baselines_training_' + problem_folder+ '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T) + '_Toy_' + '1x2'
        file_path = os.path.join('data', problem_folder, 'scenario_test1x2.json')
        energy_dist_path = os.path.join('data', problem_folder,  'energy_distance_1x2.npy')
        scenario = create_scenario(file_path, energy_dist_path)
        env = AMoD(scenario, beta=args.beta)
        gym_env = FleetEnv(env=env, gurobi_env=gurobi_env, T=args.T)
    elif not args.test:
        number_subproblems = args.number_subproblems
        training_environments = []
        training_gym_environments = []
        experiment = 'stable_baselines_training_' + problem_folder + '_' + str(number_subproblems) + '_subproblems' + '_' + str(args.max_episodes) + '_episodes_T_' + str(args.T)
        for subproblem in range(number_subproblems):
            file_path = os.path.join('data', problem_folder, 'subproblem_' + str(subproblem) + '_SF_10.json')
            energy_dist_path = os.path.join('data', problem_folder, 'subproblem_' + str(subproblem) +  '_energy_distance.npy')
            scenario = create_scenario(file_path, energy_dist_path)
            env = AMoD(scenario, beta=args.beta)
            training_environments.append(env)
            gym_env = FleetEnv(env=env, gurobi_env=gurobi_env, T=args.T)
            training_gym_environments.append(gym_env)
        env = training_environments[0]
        gym_env = training_gym_environments[0]
    else:
        experiment = 'testing_' + problem_folder + '_' + str(args.max_episodes) + '_episodes'
        file_path = os.path.join('data', problem_folder, 'SF_10.json')
        energy_dist_path = os.path.join('data', problem_folder, 'energy_distance.npy')
        test_scenario = create_scenario(file_path, energy_dist_path)
        env = AMoD(test_scenario, beta=args.beta)
        gym_env = FleetEnv(env,gurobi_env)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": args.max_episodes*env.tf,
        "forecast_horizon": args.T,
        "env_name": file_path,
    }
    run = wandb.init(
        # Set the project where this run will be logged
        project='e-amod', 
        # pass a run name 
        name=experiment, 
        # Track hyperparameters and run metadata
        config={
        "number_chargelevels": env.scenario.number_charge_levels,
        "number_spatial_nodes": env.scenario.spatial_nodes,
        "dataset": file_path,
        "episodes": args.max_episodes*env.tf,
        "rl_config": config,
        "model": "PPO",
        })

    model = SAC(policy=config["policy_type"], env=gym_env, verbose=1, train_freq=env.tf, gradient_steps=1, learning_rate=0.0001)
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


if __name__ == "__main__":
    main()
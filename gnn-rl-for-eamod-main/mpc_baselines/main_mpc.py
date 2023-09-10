import sys
import argparse
sys.path.append('./src')
from src.envs.amod_env import Scenario, AMoD #, Star2Complete
from torch_geometric.data import Data
from src.algos.a2c_gnn import A2C
from src.misc.utils import mat2str, dictsum
from MPC import MPC
import time
import os
import subprocess
from collections import defaultdict
import numpy as np
import gurobipy as gp
import json
import wandb
import pickle
import torch
import copy

def create_scenario(json_file_path, energy_file_path, seed=10):
    f = open(json_file_path)
    energy_dist = np.load(energy_file_path)
    data = json.load(f)
    tripAttr = data['demand']
    reb_time = data['rebTime']
    total_acc = data['totalAcc']
    spatial_nodes = data['spatialNodes']
    tf = data['episodeLength']
    number_charge_levels = data['chargelevels']
    charge_levels_per_charge_step = data['chargeLevelsPerChargeStep']
    chargers = data['chargeLocations']
    cars_per_station_capacity = data['carsPerStationCapacity']
    p_energy = data["energy_prices"]
    time_granularity = data["timeGranularity"]
    operational_cost_per_timestep = data['operationalCostPerTimestep']

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, cars_per_station_capacity = cars_per_station_capacity, number_charge_levels=number_charge_levels, charge_levels_per_charge_step=charge_levels_per_charge_step, 
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc, p_energy=p_energy, time_granularity=time_granularity, operational_cost_per_timestep=operational_cost_per_timestep)
    return scenario

parser = argparse.ArgumentParser(description='A2C-GNN')

parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--initial_state', type=bool, default=False,
                    help='forces MPC to return system to initial charge')
parser.add_argument('--mpc_horizon', type=int, default=36, metavar='N',
                    help='MPC horizon (default: 36)')
parser.add_argument('--subproblem', type=int, default=0, metavar='N',
                    help='which subproblem to run (default: 0)')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--spatial_nodes', type=int, default=5, metavar='N',
                    help='number of spatial nodes (default: 5)')
args = parser.parse_args()

mpc_horizon = args.mpc_horizon
num_sn = args.spatial_nodes

if args.toy:
    problem_folder = 'Toy'
    file_path = os.path.join('..', 'data', problem_folder, 'scenario_test_6_1x2_flip.json')
    experiment = file_path +  '_mpc_horizon_' + str(mpc_horizon)
    energy_dist_path = os.path.join('..', 'data', problem_folder,  'energy_distance_1x2.npy')
    scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(scenario)
    tf = env.tf
else:
    problem_folder = 'NY'
    file_path = os.path.join('..', 'data', problem_folder, str(num_sn), f'NYC_{num_sn}.json')
    # problem_folder = 'NY_5'
    # file_path = os.path.join('..', 'data', problem_folder, 'NY_5.json')
    # problem_folder = 'NY/ClusterDataset1'
    # file_path = os.path.join('..', 'data', problem_folder,  'd1.json')
    # problem_folder = 'SF_5_clustered'
    # file_path = os.path.join('..', 'data', problem_folder, 'SF_5_short_afternoon_test.json')
    # problem_folder = 'SF_10_clustered'
    # file_path = os.path.join('..', 'data', problem_folder,  'SF_10.json')
    
    experiment = problem_folder +  '_mpc_horizon_' + str(mpc_horizon) + 'entire_problem' + file_path + "_heuristic_graph"
    energy_dist_path = os.path.join('..', 'data', problem_folder, str(num_sn), 'energy_distance.npy')
    test_scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(test_scenario)
    tf = env.tf

    scenario_test = create_scenario(file_path, energy_dist_path, seed=10)
    env_test = AMoD(scenario_test)

print('mpc_horizon', mpc_horizon, 'episodeLength', tf)
experiment += "_RL_approach_constraint"

# set Gurobi environment mine
# gurobi_env = gp.Env(empty=True)
# gurobi = "Dominik"
# gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
# gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
# gurobi_env.setParam('LICENSEID', 799876)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set Gurobi environment Justin
gurobi_env = gp.Env(empty=True)
gurobi = "Justin"
gurobi_env.setParam('WLSACCESSID', '82115472-a780-40e8-9297-b9c92969b6d4')
gurobi_env.setParam('WLSSECRET', '0c069810-f45f-4920-a6cf-3f174425e641')
gurobi_env.setParam('LICENSEID', 844698)
gurobi_env.setParam("OutputFlag",0)
gurobi_env.start()

# set Gurobi environment Justin
# gurobi_env = gp.Env(empty=True)
# gurobi = "Karthik"
# gurobi_env.setParam('WLSACCESSID', 'ad632625-ffd3-460a-92a0-6fef5415c40d')
# gurobi_env.setParam('WLSSECRET', '60bd07d8-4295-4206-96e2-bb0a99b01c2f')
# gurobi_env.setParam('LICENSEID', 849913)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()

# set Gurobi environment Karthik2
# gurobi_env = gp.Env(empty=True)
# gurobi = "Karthik2"
# gurobi_env.setParam('WLSACCESSID', 'bc0f99a5-8537-45c3-89d9-53368d17e080')
# gurobi_env.setParam('WLSSECRET', '6dddd313-d8d4-4647-98ab-d6df872c6eaa')
# gurobi_env.setParam('LICENSEID', 799870)
# gurobi_env.setParam("OutputFlag",0)
# gurobi_env.start()


# set up wandb
wandb.init(
      # Set the project where this run will be logged
      project='e-amod', 
      # pass a run name 
      name=experiment, 
      # Track hyperparameters and run metadata
      config={
      "number_chargelevels": env.scenario.number_charge_levels,
      "number_spatial_nodes": env.scenario.spatial_nodes,
      "dataset": file_path,
      "mpc_horizon": mpc_horizon,
      "number_vehicles_per_node_init": env.G.nodes[(0,1)]['accInit'],
      "charging_stations": list(env.scenario.charging_stations),
      "licence": gurobi,
      })

opt_rew = []

mpc = MPC(env, gurobi_env, mpc_horizon, args.initial_state)
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0
t_0 = time.time()
time_list = []

env.reset(bool_sample_demand=False)
# print(env_test.demand)

while (not done):
    time_i_start = time.time()
    paxAction, rebAction = mpc.MPC_exact()
    time_i_end = time.time()
    t_i = time_i_end - time_i_start
    time_list.append(t_i)
    if (env.tf <= env.time + mpc_horizon):
        timesteps = range(mpc_horizon)
    else:
        timesteps = [0]
    t_reward = 0
    for t in timesteps:
        obs_1, reward1, done, info, td = env.pax_step(paxAction[t], gurobi_env)
        obs_2, reward2, done, info = env.reb_step(rebAction[t])
        
        opt_rew.append(reward1+reward2) 

        served += info['served_demand']
        rebcost += info['rebalancing_cost']
        opcost += info['operating_cost']
        revenue += info['revenue']

print(f'MPC: Reward {sum(opt_rew)}, Revenue {revenue}, Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')

# Send current statistics to wandb
wandb.log({"Reward": sum(opt_rew), "ServedDemand": served, "Reb. Cost": rebcost, "Avg.Time": np.array(time_list).mean()})

opt_rew = []
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0

for i in range(50):
    env_test.reset(bool_sample_demand=True, seed=i)
    # print(env_test.demand)

    eps_rew = []
    eps_served = []
    eps_reb = []
    eps_op = []
    eps_rev = []

    while (not done):
        if (env.tf <= env.time + mpc_horizon):
            timesteps = range(mpc_horizon)
        else:
            timesteps = [0]
        
        for t in timesteps:
            obs_1, reward1, done, info, td = env_test.pax_step(paxAction[t], gurobi_env)
            obs_2, reward2, done, info = env_test.reb_step(rebAction[t])

            eps_rew.append(reward1+reward2)
            eps_served.append(info['served_demand'])
            eps_reb.append(info['rebalancing_cost'])
            eps_op.append(info['operating_cost'])
            eps_rev.append(info['revenue'])
            
    opt_rew.append(sum(eps_rew)) 
    served += sum(eps_served)
    rebcost += sum(eps_reb)
    opcost += sum(eps_op)
    revenue += sum(eps_rev)

opt_rew = np.mean(np.array(opt_rew))
served = np.mean(np.array(served))
rebcost = np.mean(np.array(rebcost))
opcost = np.mean(np.array(opcost))
revenue = np.mean(np.array(revenue))

print(f'Test: Reward {opt_rew}, Revenue {revenue}, Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')

# Send current statistics to wandb
wandb.log({"Test Reward": sum(opt_rew), "Test ServedDemand": served, "Test Reb. Cost": rebcost, "Avg.Time": np.array(time_list).mean()})

with open(f"./saved_files/ckpt/{problem_folder}/acc.p", "wb") as file:
    pickle.dump(env.acc, file)

wandb.save(f"./saved_files/ckpt/{problem_folder}/acc.p")
wandb.save(f"./saved_files/ckpt/{problem_folder}/satisfied_demand.p")
wandb.finish()

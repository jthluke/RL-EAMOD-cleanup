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

parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--initial_state', type=bool, default=False,
                    help='forces MPC to return system to initial charge')
parser.add_argument('--mpc_horizon', type=int, default=60, metavar='N',
                    help='MPC horizon (default: 60)')
parser.add_argument('--subproblem', type=int, default=0, metavar='N',
                    help='which subproblem to run (default: 0)')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
args = parser.parse_args()

mpc_horizon = args.mpc_horizon

if args.toy:
    problem_folder = 'Toy'
    file_path = os.path.join('..', 'data', problem_folder, 'scenario_test_6_1x2_flip.json')
    experiment = file_path +  '_mpc_horizon_' + str(mpc_horizon)
    energy_dist_path = os.path.join('..', 'data', problem_folder,  'energy_distance_1x2.npy')
    scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(scenario)
    tf = env.tf
else:
    problem_folder = 'NY_5'
    file_path = os.path.join('..', 'data', problem_folder, 'NY_5.json')
    # problem_folder = 'NY/ClusterDataset1'
    # file_path = os.path.join('..', 'data', problem_folder,  'd1.json')
    # problem_folder = 'SF_5_clustered'
    # file_path = os.path.join('..', 'data', problem_folder, 'SF_5_short_afternoon_test.json')
    # problem_folder = 'SF_10_clustered'
    # file_path = os.path.join('..', 'data', problem_folder,  'SF_10.json')
    
    experiment = problem_folder +  '_mpc_horizon_' + str(mpc_horizon) + 'entire_problem' + file_path + "_heuristic_graph"
    energy_dist_path = os.path.join('..', 'data', problem_folder, 'energy_distance.npy')
    test_scenario = create_scenario(file_path, energy_dist_path)
    env = AMoD(test_scenario)
    tf = env.tf
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

class GNNParser():
    """
    Parser converting raw environment observations to agent input.
    """

    def __init__(self, env, T=10, scale_factor=0.01, scale_price=0.1, input_size=20):
        super().__init__()
        self.env = env
        self.T = T
        self.scale_factor = scale_factor
        self.price_scale_factor = scale_price
        self.input_size = input_size

    def parse_obs(self, obs):
        # nodes
        x = torch.cat((
            torch.tensor([float(n[1])/self.env.scenario.number_charge_levels for n in self.env.nodes]).view(1, 1, self.env.number_nodes).float(),
            torch.tensor([obs[0][n][self.env.time+1] * self.scale_factor for n in self.env.nodes]).view(1, 1, self.env.number_nodes).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.scale_factor for n in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes).float(),
            torch.tensor([[sum([self.env.price[o[0], j][t]*self.scale_factor*self.price_scale_factor*(self.env.demand[o[0], j][t])*((o[1]-self.env.scenario.energy_distance[o[0], j]) >= int(not self.env.scenario.charging_stations[j]))
                          for j in self.env.region]) for o in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes).float()),
                      dim=1).squeeze(0).view(2+self.T + self.T, self.env.number_nodes).T
        # edges
        edges = []
        for o in self.env.nodes:
            for d in self.env.nodes:
                if (o[0] == d[0] and o[1] == d[1]):
                    edges.append([o, d])
        edge_idx = torch.tensor([[], []], dtype=torch.long)
        for e in edges:
            origin_node_idx = self.env.nodes.index(e[0])
            destination_node_idx = self.env.nodes.index(e[1])
            new_edge = torch.tensor([[origin_node_idx], [destination_node_idx]], dtype=torch.long)
            edge_idx = torch.cat((edge_idx, new_edge), 1)
        edge_index = torch.cat((edge_idx, self.env.gcn_edge_idx), 1)
        # print(self.env.gcn_edge_idx.shape)
        # print(edge_idx.shape)
        data = Data(x, edge_index)
        return data

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
# obs = env.reset(bool_sample_demand=random_dem) # TODO: determine if we should do this
mpc = MPC(env, gurobi_env, mpc_horizon, args.initial_state)
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0
t_0 = time.time()
time_list = []
SARS = {}
# print(len(env.nodes))
# print(len(env.nodes[0]))
# print(env.number_nodes)

if args.test:
    test_episodes = 50

while(not done):
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
    
    if args.test:
        for episode in range(test_episodes):
            env.reset()
            for t in timesteps:
                if t > 0:
                    obs1 = copy.deepcopy(o)

                obs_1, reward1, done, info, td = env.pax_step(paxAction[t], gurobi_env)
                o = GNNParser(env).parse_obs(obs_1)

                t_reward += reward1
                if t > 0: 
                    rew = (reward1 + reward2)
                    
                    action = [0 for i in range(env.number_nodes)]
                    acc, _, dacc, demand = obs_2
                    total_vehicles = sum(acc[env.nodes[i]][0] for i in range(env.number_nodes))
                    for i in range(env.number_nodes):
                        action[i] = acc[env.nodes[i]][1]/total_vehicles

                    SARS[t] = [obs1, action, rew, o]
                
                obs_2, reward2, done, info = env.reb_step(rebAction[t])
                
                opt_rew.append(reward1+reward2) 

                served += info['served_demand']
                rebcost += info['rebalancing_cost']
                opcost += info['operating_cost']
                revenue += info['revenue']
        served = served/50
        rebcost = rebcost/50
        opcost = opcost/50
        revenue = revenue/50
    else:
        for t in timesteps:
            if t > 0:
                obs1 = copy.deepcopy(o)

            obs_1, reward1, done, info, td = env.pax_step(paxAction[t], gurobi_env)
            o = GNNParser(env).parse_obs(obs_1)

            t_reward += reward1
            if t > 0: 
                rew = (reward1 + reward2)
                
                action = [0 for i in range(env.number_nodes)]
                acc, _, dacc, demand = obs_2
                total_vehicles = sum(acc[env.nodes[i]][0] for i in range(env.number_nodes))
                for i in range(env.number_nodes):
                    action[i] = acc[env.nodes[i]][1]/total_vehicles

                SARS[t] = [obs1, action, rew, o]
            
            obs_2, reward2, done, info = env.reb_step(rebAction[t])
            
            opt_rew.append(reward1+reward2) 

            served += info['served_demand']
            rebcost += info['rebalancing_cost']
            opcost += info['operating_cost']
            revenue += info['revenue'] 

if not args.test:
    print(f'MPC: Reward {sum(opt_rew)}, Revenue {revenue}, Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')
else:
    print(f'MPC: Reward {sum(opt_rew)/test_episodes}, Avg Revenue {revenue}, Avg Served demand {served}, Avg Rebalancing Cost {rebcost}, Avg Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')

# Send current statistics to wandb
wandb.log({"Reward": sum(opt_rew), "ServedDemand": served, "Reb. Cost": rebcost})
wandb.log({"Reward": sum(opt_rew), "ServedDemand": served, "Reb. Cost": rebcost, "Avg.Time": np.array(time_list).mean()})

# with open("MPC_SARS.pkl", "wb") as f:
#     pickle.dump(SARS, f)
with open(f"./saved_files/ckpt/{problem_folder}/acc.p", "wb") as file:
    pickle.dump(env.acc, file)
wandb.save(f"./saved_files/ckpt/{problem_folder}/acc.p")
wandb.save(f"./saved_files/ckpt/{problem_folder}/satisfied_demand.p")
wandb.finish()
wandb.finish()

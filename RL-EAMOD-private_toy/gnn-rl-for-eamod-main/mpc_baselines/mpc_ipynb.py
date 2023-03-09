import sys
sys.path.insert(0, '../')
from src.envs.amod_env import Scenario, AMoD #, Star2Complete
from src.misc.utils import mat2str, dictsum
from mpc_baselines.MPC import MPC
import time
import os
import subprocess
from collections import defaultdict
import numpy as np
import gurobipy as gp
import json

# MPC exact
number_nodes = 5
number_charge_levels = 5
chargers = []
for node in range(number_nodes):
    if node%2 ==1:
        chargers.append(node)
f = open('../data/scenario_test1x10.json')
data = json.load(f)
tripAttr = data["demand"]
reb_time = data["rebTime"]
total_acc = data["totalAcc"]

scenario = Scenario(charging_stations=chargers, number_charge_levels=number_charge_levels, sd=10, tripAttr = tripAttr, demand_ratio=1, reb_time=reb_time, total_acc = total_acc)
env = AMoD(scenario, beta=0.5)

# set Gurobi environment
gurobi_env = gp.Env(empty=True)
gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
gurobi_env.setParam('LICENSEID', 799876)
gurobi_env.setParam("OutputFlag",0)
gurobi_env.start()

mpc = MPC(env, gurobi_env)
opt_rew = []
obs = env.reset()
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0
t_0 = time.time()
time_list = []
while(not done):
    time_i_start = time.time()
    paxAction, rebAction = mpc.MPC_exact() 
    time_i_end = time.time()
    t_i = time_i_end - time_i_start
    time_list.append(t_i)
    
    obs, reward1, done, info = env.pax_step(paxAction)
    
    obs, reward2, done, info = env.reb_step(rebAction)
    opt_rew.append(reward1+reward2) 
    served += info['served_demand']
    rebcost += info['rebalancing_cost']
    opcost += info['operating_cost']
    revenue += info['revenue'] 
print(f'MPC: Reward {sum(opt_rew)}, Revenue {revenue},Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.2f} +- {np.array(time_list).std():.2f}sec')
demand = sum([env.demand[i,j][t] for i,j in env.demand for t in range(0,60)])
print(demand, served/demand)
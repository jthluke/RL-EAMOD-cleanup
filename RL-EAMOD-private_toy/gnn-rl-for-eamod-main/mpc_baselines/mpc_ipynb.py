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

# set Gurobi environment Karthik2
gurobi_env = gp.Env(empty=True)
gurobi = "Karthik2"
gurobi_env.setParam('WLSACCESSID', 'bc0f99a5-8537-45c3-89d9-53368d17e080')
gurobi_env.setParam('WLSSECRET', '6dddd313-d8d4-4647-98ab-d6df872c6eaa')
gurobi_env.setParam('LICENSEID', 799870)
gurobi_env.setParam("OutputFlag",0)
gurobi_env.start()

# MPC exact
scenario = Scenario(json_file="data/scenario_nyc4x4.json", sd=10, demand_ratio=.5, json_hr=7,json_tstep=3)
env = AMoD(scenario, beta=0.5)

mpc = MPC(env, CPLEXPATH,T=10)
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
#     print(f"t_i: {t_i:.1f}sec")
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

# three levels
# scenario = Star2Complete(sd = 10, grid_travel_time = 2, T = 16, star_demand = 7.5, complete_demand=1.5, 
#                         star_center = [9,10,13,14], beta=0.9, alpha = 0.5, ninit = 200, fix_price = True)

scenario = Scenario(json_file="data/scenario_nyc4x4.json", sd=10, demand_ratio=.5, json_hr=7,json_tstep=3)
env = AMoD(scenario, beta=0.5)

mpc = MPC(env, CPLEXPATH,T=10)
opt_rew = []
obs = env.reset()
done = False
served = 0
rebcost = 0
opcost = 0
revenue = 0
paxFlow = dict()
rebFlow = dict()
desiredAcc = dict()
rebAction = dict()
t_0 = time.time()
time_list = []
action_list = []
for _ in range(100):
    try:
        while(not done):
            t = env.time
            time_i_start = time.time()
            res_path = 'tri-level/'
            desiredAcc[t], paxFlow[t], rebFlow[t] = mpc.tri_level()
            action_list.append(list(desiredAcc[t].values()))
            time_i_end = time.time()
            t_i = time_i_end - time_i_start
            time_list.append(t_i)
            print(f"t_i: {t_i:.1f}sec")
            obs, reward1, done, info = env.pax_step(CPLEXPATH = CPLEXPATH, PATH = res_path)

            rebAction[t] = solveRebFlow(env,'reb_'+res_path,desiredAcc[t],CPLEXPATH)
            obs, reward2, done, info = env.reb_step(rebAction[t])
            opt_rew.append(reward1+reward2) 
            served += info['served_demand']
            rebcost += info['rebalancing_cost']
            opcost += info['operating_cost']
            revenue += info['revenue']
    except:
        continue
print(f'Downgraded MPC (three levels): Reward {sum(opt_rew)}, Revenue {revenue},Served demand {served}, Rebalancing Cost {rebcost}, Operational Cost {opcost}, Avg.Time: {np.array(time_list).mean():.1f} +- {np.array(time_list).std():.1f}sec')
demand = sum([env.demand[i,j][t] for i,j in env.demand for t in range(0,60)])
print(demand, served/demand)
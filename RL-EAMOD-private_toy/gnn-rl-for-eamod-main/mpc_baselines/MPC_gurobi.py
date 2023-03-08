from socketserver import ThreadingUnixDatagramServer
import gurobipy as gp
import numpy as np
import math
import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str
import copy

def solve_mpc(env, gurobi_env=None, mpc_horizon=30):
    time = env.time
    discount_factor = 0.99
    if mpc_horizon+time >= env.tf:
        discount_factor = 1
    m = gp.Model(env=gurobi_env)
    dacc = defaultdict(dict) # should be all zeros at the start
    acc = defaultdict(dict)
    for n in env.nodes:
        dacc[n] = defaultdict(int)
        acc[n] = defaultdict(int)
        acc[n][0] = env.acc[n][time]
        for t in range(int(mpc_horizon*2)):
            dacc[n][t] = 0
    pax_flow = m.addMVar(shape=(mpc_horizon, len(env.edges)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="pax_flow")
    rebal_flow = m.addMVar(shape=(mpc_horizon, len(env.edges)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="rebal_flow")
    charging_cars_per_location = defaultdict(dict)
    for n in env.nodes_spatial:
        charging_cars_per_location[n] = defaultdict(float)
        for t in range(mpc_horizon):
            charging_cars_per_location[n][t] = copy.copy(env.scenario.cars_charging_per_station[n][t+time+1])
    for t in range(mpc_horizon):
        for o in env.region:
            for d in env.region:
                # Constraint: no more passenger flow than demand on pax edges
                m.addConstr(
                    sum(pax_flow[t, env.map_o_d_regions_to_pax_edges[(o,d)]]) <= env.demand[o, d][t + time]
                )
        # pax flow should be zero on rebal edges
        m.addConstr(
                    sum(pax_flow[t, env.charging_edges]) == 0
                )
        for n in env.nodes:
            outgoing_edges = env.map_node_to_outgoing_edges[n]
            
            # Constraint: We can not have more vehicles flowing out of a node, than vehicles at the node
            m.addConstr(
                sum(rebal_flow[t, outgoing_edges]) +  sum(pax_flow[t, outgoing_edges]) <= acc[n][t] 
            )
            for e in outgoing_edges:
                o_node, d_node = env.edges[e]
                # add incoming flow from earlier optimizations -> should be stored in rebal_flow and pax_flow
                # only relevant if we do not solve everything as one optimization
                if mpc_horizon < env.tf:
                    if env.rebFlow[o_node,d_node][t + time] != 0.0:
                        dacc[d_node][t+1] += env.rebFlow[o_node,d_node][t + time]
                    if env.paxFlow[o_node,d_node][t + time] != 0.0:
                        dacc[d_node][t+1] += env.paxFlow[o_node,d_node][t + time]
                # regular opimization
                dacc[d_node][t+env.G.edges[o_node,d_node]['time'][t + time]] += pax_flow[t,e] + rebal_flow[t,e] # adding one because of design decision to appear later
                # charge station constraint
                if d_node[1] > o_node[1] or (o_node[1] - env.scenario.energy_distance[o_node[0], d_node[0]] < d_node[1] and o_node[0] != d_node[0]):
                    # Constraint: no more charging vehicles than there are charging stations
                    m.addConstr(
                        charging_cars_per_location[o_node[0]][t] + rebal_flow[t,e] <= env.scenario.cars_per_station_capacity[o_node[0]]
                    )

                    if d_node[1] > o_node[1] and o_node[0] == d_node[0]:
                        change_dist = d_node[1] - o_node[1]
                    else:
                        change_dist = d_node[1] - o_node[1] + env.scenario.energy_distance[o_node[0], d_node[0]]
                    
                    time_spent_charging = math.ceil(change_dist/env.scenario.charge_levels_per_charge_step)
                    # add plus one because range creates values from [begin,end)
                    for future_time_step in range(t,t+time_spent_charging):
                        charging_cars_per_location[o_node[0]][future_time_step] = charging_cars_per_location[o_node[0]][future_time_step] + rebal_flow[t,e] 
            
        for n in env.nodes:
            outgoing_edges = env.map_node_to_outgoing_edges[n]
            acc[n][t+1] = acc[n][t] + dacc[n][t] - sum(rebal_flow[t, outgoing_edges]) - sum(pax_flow[t, outgoing_edges]) 


    obj = 0
    for t in range(mpc_horizon):
        for e in range(len(env.edges)):
            o_node,d_node = env.edges[e]
            o_region = o_node[0]
            d_region = d_node[0]
            charge_cost = 0
            charge_time = 0
            if d_node[1]>=o_node[1] and o_region != o_region:
                charge_diff = d_node[1] - o_node[1] + env.scenario.energy_distance[o_region,d_region]
                charge_time = math.ceil(charge_diff/env.scenario.charge_levels_per_charge_step)
                avg_price = np.mean(env.scenario.p_energy[time+t:time+t+charge_time])
                charge_cost = avg_price*charge_diff*rebal_flow[t,e]
            elif d_node[1]>o_node[1]:
                charge_diff = d_node[1] - o_node[1]
                charge_time = math.ceil(charge_diff/env.scenario.charge_levels_per_charge_step)
                avg_price = np.mean(env.scenario.p_energy[time+t:time+t+charge_time])
                charge_cost = avg_price*charge_diff*rebal_flow[t,e]
            obj += (discount_factor**t) * pax_flow[t,e]*env.price[o_region,d_region][t + time] - env.scenario.operational_cost_per_timestep * (rebal_flow[t,e] + pax_flow[t,e]) * (env.G.edges[o_node,d_node]['time'][t + time]+env.scenario.time_normalizer - charge_time) - charge_cost
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    assert m.status == 2
    pax = pax_flow.X
    rebal = rebal_flow.X
    
    return pax,rebal



# Class def for optimization
import gurobipy as gp
from gurobipy import quicksum
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import os
import time

def compute_obj_component(i, flow, edges, price, G, current_time, stn, ocpt):
    return flow[i] * (price[edges[i][0][0], edges[i][1][0]][current_time] - (G.edges[edges[i]]['time'][current_time] + stn) * ocpt)

class PaxFlowsSolver:

    def __init__(self, env, gurobi_env):
        # Initialize model
        self.env = env
        self.cons_charge_graph = {}
        self.cons_spatial_graph = {}
        self.cons_rebal_edges = {}
        t = self.env.time
        self.m = gp.Model(env=gurobi_env)

        self.m.Params.Method = 2
        self.m.Params.Crossover = 0
        self.m.Params.BarConvTol = 1e-6
        self.m.Params.Threads = 60
        self.m.setParam("LogFile", os.path.join(os.getcwd(), 'pax_flow_gurobi_log.log'))

        self.flow = self.m.addMVar(shape=(len(
            self.env.edges)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="flow")

        # constr. 1: can't have outflow more than initial accumulation
        for n in self.env.nodes:
            self.cons_charge_graph[n] = self.m.addConstr(
                sum(self.flow[self.env.map_node_to_outgoing_edges[n]]) <= float(
                    self.env.acc[n][t])
            )

        # constr. 2: no more flow than demand
        for i in self.env.region:
            for j in self.env.region:
                self.cons_spatial_graph[(i, j)] = self.m.addConstr(
                    sum(self.flow[self.env.map_o_d_regions_to_pax_edges[(
                        i, j)]]) <= self.env.demand[i, j][t]
                )
        # constr. 3: pax flow is zero on rebal edges
        self.cons_rebal_edges[0] = self.m.addConstr(
            sum(self.flow[self.env.charging_edges]) == 0
        )
        # objective function: maximize profit
        obj = 0
        for i in range(len(self.env.edges)):
            edge = self.env.edges[i]
            o_region = edge[0][0]
            d_region = edge[1][0]
            obj += self.flow[i] * (self.env.price[o_region, d_region][t] - (self.env.G.edges[edge]['time']
                                   [self.env.time]+self.env.scenario.time_normalizer) * self.env.scenario.operational_cost_per_timestep)
        self.m.setObjective(obj, gp.GRB.MAXIMIZE)

    def update_constraints(self):
        for n in self.env.nodes:
            self.cons_charge_graph[n].RHS = float(
                self.env.acc[n][self.env.time])
        for i in self.env.region:
            for j in self.env.region:
                self.cons_spatial_graph[(
                    i, j)].RHS = self.env.demand[i, j][self.env.time]
        self.m.update()

    def update_objective(self):
        time_a = time.time()

        stn = self.env.scenario.time_normalizer
        ocpt = self.env.scenario.operational_cost_per_timestep
        current_time = self.env.time

        # Create a pool of workers
        pool = Pool()

        # Use the pool's map function to parallelize the computation
        results = pool.map(
            compute_obj_component, 
            range(len(self.env.edges)), 
            [self.flow]*len(self.env.edges), 
            [self.env.edges]*len(self.env.edges), 
            [self.env.price]*len(self.env.edges), 
            [self.env.G]*len(self.env.edges), 
            [current_time]*len(self.env.edges), 
            [stn]*len(self.env.edges), 
            [ocpt]*len(self.env.edges)
        )

        # Sum up the results
        obj = sum(results)
        
        time_a_end = time.time() - time_a

        time_b = time.time()
        self.m.setObjective(obj, gp.GRB.MAXIMIZE)
        time_b_end = time.time() - time_b

        time_c = time.time()
        self.m.update()
        time_c_end = time.time() - time_c

        print(f"Time: {time_a_end}, {time_b_end}, {time_c_end}")

    # def optimize(self):
    #     self.m.optimize()
    #     paxAction = self.flow.X
    #     return paxAction
    
    def optimize(self):
        self.m.optimize()
        if self.m.status == 3:
            print("Optimization is infeasible.")
            # Return a default flow
            return np.zeros(self.flow.shape)
        elif self.m.status != 2:
            print("Optimization did not complete successfully.")
            return np.zeros(self.flow.shape)  # or handle other statuses as needed
        paxAction = self.flow.X
        return paxAction

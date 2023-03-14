# Class def for optimization
import gurobipy as gp


class RebalFlowSolver:
    def __init__(self, env, desired_acc_spatial, gurobi_env=None):
        # Initialize model
        self.cons_charge_graph = {}
        self.cons_spatial_graph = {}
        t = env.time
        # start_time = os.times()
        self.m = gp.Model(env=gurobi_env)
        self.flow = self.m.addMVar(shape=(len(env.edges)), lb=0.0, ub=gp.GRB.INFINITY,
                                   vtype=gp.GRB.CONTINUOUS, name="flow")  # both could be INTEGER
        
        acc_spatial_with_charge = dict()
        for n in env.nodes_spatial:
            acc_spatial_with_charge[n] = env.acc_spatial[n][env.time + 1]
            for c in range(env.scenario.max_energy_distance):
                acc_spatial_with_charge[n] -= env.acc[(n,c)][env.time + 1]
        
        # should be unnecessary:
        for e_idx in range(len(env.edges)):
            self.m.addConstr(self.flow[e_idx] >= 0.)
        for e_charging in env.charging_edges:
            self.m.addConstr(self.flow[e_charging] == 0.)

        for n_idx in range(len(env.nodes)):
            n = env.nodes[n_idx]
            outgoing_edges = env.map_node_to_outgoing_edges[n]

            # Constraint 0: We can not have more vehicles flowing out of a node, than vehicles at the node
            self.cons_charge_graph[n_idx] = self.m.addConstr(
                sum(self.flow[outgoing_edges]) <= env.acc[n][t + 1])
        for node_spatial in env.nodes_spatial:
            corresponing_road_edges_charge_graph_incoming = env.map_node_spatial_to_incoming_road_edges_charge_graph[
                node_spatial]
            corresponing_road_edges_charge_graph_outgoing = env.map_node_spatial_to_outgoing_road_edges_charge_graph[
                node_spatial]
            # Constraint1: should only use road edges to rebalance idle vehicles to match target distribution.
            self.cons_spatial_graph[node_spatial] = self.m.addConstr(sum(self.flow[corresponing_road_edges_charge_graph_incoming]) - sum(self.flow[corresponing_road_edges_charge_graph_outgoing]) ==
                desired_acc_spatial[node_spatial] - acc_spatial_with_charge[node_spatial]) 
        self.obj = 0
        for e_idx in range(len(env.edges)):
            i, j = env.edges[e_idx]
            self.obj += self.flow[e_idx] * (env.G.edges[i, j]['time']
                                             [t + 1] + env.scenario.time_normalizer)* env.scenario.operational_cost_per_timestep
        self.m.setObjective(self.obj, gp.GRB.MINIMIZE)
        # end_time = os.times()
        # print("Time for creating model: ", end_time[0] - start_time[0])
        self.env = env

    def update_constraints(self, desired_acc_spatial, env):
        acc_spatial_with_charge = dict()
        for n in env.nodes_spatial:
            acc_spatial_with_charge[n] = env.acc_spatial[n][env.time + 1]
            for c in range(env.scenario.max_energy_distance):
                acc_spatial_with_charge[n] -= env.acc[(n,c)][env.time + 1]
        test_sum_desired = 0
        test_sum_acc = 0
        for node_spatial in env.nodes_spatial:
            test_sum_desired += desired_acc_spatial[node_spatial]
            test_sum_acc += acc_spatial_with_charge[node_spatial]
            self.cons_spatial_graph[node_spatial].RHS = desired_acc_spatial[node_spatial] - acc_spatial_with_charge[node_spatial]
        assert abs(test_sum_desired - test_sum_acc) < 0.00001
        for n_idx in range(len(env.nodes)):
            node_charge = env.nodes[n_idx]
            self.cons_charge_graph[n_idx].RHS = env.acc[node_charge][env.time + 1]
        self.m.update()

    def update_objective(self, env):
        self.obj = sum((self.flow[e_idx] * (env.G.edges[env.edges[e_idx][0], env.edges[e_idx][1]]
                        ['time'][env.time + 1] + env.scenario.time_normalizer) * env.scenario.operational_cost_per_timestep) for e_idx in range(len(env.edges)))
        self.m.setObjective(self.obj, gp.GRB.MINIMIZE)
        self.m.update()

    def optimize(self):
        self.m.optimize()
        if self.m.status == 3:
            print("Optimization is infeasible.")
        assert self.m.status == 2
        action = self.flow.X
        return action

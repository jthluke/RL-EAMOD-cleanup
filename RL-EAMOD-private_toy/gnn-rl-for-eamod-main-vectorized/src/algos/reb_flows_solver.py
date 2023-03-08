# Class def for optimization
import gurobipy as gp

class RebalFlowSolver:  
    def __init__(self, gurobi_para=None):
        # Initialize model
        self.gurobi_para = gurobi_para
        
    def update_model_and_optimize(self, desired_acc, env):
        gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam('WLSACCESSID', self.gurobi_para['WLSACCESSID'])
        gurobi_env.setParam('WLSSECRET', self.gurobi_para['WLSSECRET'])
        gurobi_env.setParam('LICENSEID', self.gurobi_para['LICENSEID'])
        gurobi_env.setParam("OutputFlag",0)
        gurobi_env.start()
        t = env.time
        m = gp.Model(env=gurobi_env)
        flow = m.addMVar(shape=(len(env.edges)), lb=0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="flow") # both could be INTEGER
        slack_variables = m.addMVar(shape=(len(env.nodes)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="slack")

        for n_idx in range(len(env.nodes)):
            n = env.nodes[n_idx]
            outgoing_edges = env.map_node_to_outgoing_edges[n]
            incoming_edges = env.map_node_to_incoming_edges[n]

            # Constraint 0: We can not have more vehicles flowing out of a node, than vehicles at the node
            m.addConstr(sum(flow[outgoing_edges]) <= int(env.acc[n][t + 1]))
            # Constrain 1: We want to reach the target distribrution
            m.addConstr(sum(flow[incoming_edges]) - sum(flow[outgoing_edges]) + slack_variables[n_idx] >= int(desired_acc[n]) - int(env.acc[n][t + 1]) )
        obj1 = 0
        obj2 = 0
        for n_idx in range(len(env.nodes)):
            obj1 += slack_variables[n_idx] * 1e10

        for e_idx in range(len(env.edges)):
            i,j = env.edges[e_idx]
            obj2 += flow[e_idx] * (env.G.edges[i,j]['time'][t + 1]+env.scenario.time_normalizer) * env.scenario.operational_cost_per_timestep
        m.setObjective(obj1+obj2, gp.GRB.MINIMIZE)
        m.optimize()
        assert m.status==2
        action = flow.X
        return action
        
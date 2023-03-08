# Class def for optimization
import gurobipy as gp


class PaxFlowsSolver:

    def __init__(self, gurobi_para):
        # Initialize model
        self.gurobi_para = gurobi_para
        

    def update_model_and_optimize(self, env):
        t = env.time
        gurobi_env = gp.Env(empty=True)
        gurobi_env.setParam('WLSACCESSID', self.gurobi_para['WLSACCESSID'])
        gurobi_env.setParam('WLSSECRET', self.gurobi_para['WLSSECRET'])
        gurobi_env.setParam('LICENSEID', self.gurobi_para['LICENSEID'])
        gurobi_env.setParam("OutputFlag",0)
        gurobi_env.start()
        m = gp.Model(env=gurobi_env)
        flow = m.addMVar(shape=(len(
            env.edges)), lb=0.0, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="flow")

        # constr. 1: can't have outflow more than initial accumulation
        for n in env.nodes:
            m.addConstr(
                sum(flow[env.map_node_to_outgoing_edges[n]]) <= float(
                    env.acc[n][t])
            )

        # constr. 2: no more flow than demand
        for i in env.region:
            for j in env.region:
                m.addConstr(
                    sum(flow[env.map_o_d_regions_to_pax_edges[(
                        i, j)]]) <= env.demand[i, j][t]
                )
        # constr. 3: pax flow is zero on rebal edges
        m.addConstr(
            sum(flow[env.charging_edges]) == 0
        )
        # objective function
        obj = 0
        for i in range(len(env.edges)):
            edge = env.edges[i]
            o_region = edge[0][0]
            d_region = edge[1][0]
            obj += flow[i] * (env.price[o_region, d_region][t] - (env.G.edges[edge]['time']
                                   [env.time]+env.scenario.time_normalizer) * env.scenario.operational_cost_per_timestep)
        m.setObjective(obj, gp.GRB.MAXIMIZE)
        m.optimize()
        paxAction = flow.X
        return paxAction

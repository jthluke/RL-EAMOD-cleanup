import wandb
from gym import spaces
import gym
from torch.distributions import Dirichlet
import torch.nn.functional as F
import torch
from src.algos.reb_flows_solver import RebalFlowSolver
from src.algos.pax_flows_solver import PaxFlowsSolver
from cmath import inf
import numpy as np
import sys
sys.path.insert(0, '../../')


class FleetEnv(gym.Env):
    """
    Custom Environment that follows gym interface. 
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, env=None, gurobi_para=None, T=10, scale_factor_reward=0.01, scale_factor=0.0001, scale_factor_price=0.1, experiment=None, rank=True):
        super(FleetEnv, self).__init__()
        self.rank = rank
        self.experiment = experiment
        self.env = env
        self.episode = 0
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_rebalancing_cost = 0
        self.T = T
        self.pax_flows_solver = PaxFlowsSolver(gurobi_para=gurobi_para)
        desired_acc = {env.nodes[i]: int(self.env.G.nodes[env.nodes[i]]['accInit'])
                       for i in range(env.number_nodes)}  # over nodes
        pax_flow = self.pax_flows_solver.update_model_and_optimize(env)
        _, paxreward, done, info_pax = env.pax_step(
            paxAction=pax_flow)
        self.episode_served_demand += info_pax['served_demand']
        self.episode_reward += paxreward
        # definition rebal solver
        self.rebal_flow_solver = RebalFlowSolver(gurobi_para=gurobi_para)
        # Define action and observation space
        self.scale_factor = scale_factor
        self.scale_factor_price = scale_factor_price
        self.scale_factor_reward = scale_factor_reward
        self.initial_state = self.parse_state().astype(np.float32)

        self.action_space = spaces.Box(
            low=0, high=1000, shape=(len(self.env.nodes)*2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=self.initial_state.shape, dtype=np.float32)

        if self.rank == 0:
            self.run = wandb.init(
            # Set the project where this run will be logged
            project='e-amod',
            # pass a run name
            name=self.experiment,
            sync_tensorboard=True,
            # Track hyperparameters and run metadata
            config={
                "number_chargelevels": self.env.scenario.number_charge_levels,
                "number_spatial_nodes": self.env.scenario.spatial_nodes,
                "number_vehicles_per_node_init": self.env.G.nodes[(0, int(self.env.scenario.number_charge_levels*0.4))]['accInit'],
                "charging_stations": list(self.env.scenario.charging_stations),
                "charging_station_capacities": list(self.env.scenario.cars_per_station_capacity),
                "scale_factor": self.scale_factor,
                "scale_price": self.scale_factor_price,
                "episode_length": self.env.tf,
                "charge_levels_per_timestep": self.env.scenario.charge_levels_per_charge_step,
            })

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        if self.rank == 0:
            wandb.log({"Episode": self.episode, "Reward": self.episode_reward,
                      "ServedDemand": self.episode_served_demand, "Reb. Cost": self.episode_rebalancing_cost})
        self.episode += 1
        self.episode_reward = 0
        self.episode_served_demand = 0
        self.episode_rebalancing_cost = 0
        self.env.reset()
        pax_flow = self.pax_flows_solver.update_model_and_optimize(self.env)
        _, paxreward, done, info_pax = self.env.pax_step(
            paxAction=pax_flow)
        self.episode_served_demand += info_pax['served_demand']
        self.episode_reward += paxreward
        return self.initial_state

    def step(self, action):
        # print("action", action.shape)
        a_out_concentration = action[:len(self.env.nodes)]
        a_out_is_zero = action[len(self.env.nodes):]
        jitter = 1e-20
        concentration = F.softplus(torch.tensor(
            a_out_concentration)).reshape(-1) + jitter
        non_zero = torch.sigmoid(torch.tensor(a_out_is_zero)).reshape(-1)

        concentration_without_zeros = torch.tensor([], dtype=torch.float32)
        sampled_zero_bool_arr = []
        log_prob_for_zeros = 0
        for node in range(non_zero.shape[0]):
            sample = torch.bernoulli(non_zero[node])
            if sample > 0:
                indices = torch.tensor([node])
                new_element = torch.index_select(concentration, 0, indices)
                concentration_without_zeros = torch.cat(
                    (concentration_without_zeros, new_element), 0)
                sampled_zero_bool_arr.append(False)
                log_prob_for_zeros += torch.log(non_zero[node])
            else:
                sampled_zero_bool_arr.append(True)
                log_prob_for_zeros += torch.log(1-non_zero[node])
        if concentration_without_zeros.shape[0] != 0:
            m = Dirichlet(concentration_without_zeros)
            dirichlet_action = m.rsample()
            dirichlet_action_np = list(dirichlet_action.detach().numpy())
        action_np = []
        dirichlet_idx = 0
        for node in range(non_zero.shape[0]):
            if sampled_zero_bool_arr[node]:
                action_np.append(0.)
            else:
                action_np.append(dirichlet_action_np[dirichlet_idx])
                dirichlet_idx += 1

        total_idle_acc = sum(
            self.env.acc[n][self.env.time+1] for n in self.env.nodes)
        desired_acc = {self.env.nodes[i]: int(
            action_np[i] * total_idle_acc) for i in range(self.env.number_nodes)}  # over nodes
        total_desiredAcc = sum(desired_acc[n] for n in self.env.nodes)
        missing_cars = total_idle_acc - total_desiredAcc
        most_likely_node = np.argmax(action_np)
        if missing_cars != 0:
            desired_acc[self.env.nodes[most_likely_node]] += missing_cars
            total_desiredAcc = sum(desired_acc[n] for n in self.env.nodes)
        assert abs(total_desiredAcc - total_idle_acc) < 1e-5
        for n in self.env.nodes:
            assert desired_acc[n] >= 0
        # solve minimum rebalancing distance problem (Step 3 in paper)
        rebAction = self.rebal_flow_solver.update_model_and_optimize(desired_acc, self.env)

        _, reb_reward, done, info = self.env.reb_step(rebAction)
        self.episode_rebalancing_cost += info['rebalancing_cost']
        self.episode_reward += reb_reward
        reward = reb_reward
        if not done:
            pax_flow = self.pax_flows_solver.update_model_and_optimize(self.env)
            _, paxreward, done, info_pax = self.env.pax_step(
                paxAction=pax_flow)
            self.episode_served_demand += info_pax['served_demand']
            self.episode_reward += paxreward
            reward += paxreward
            state = self.parse_state()
        else:
            state = self.initial_state

        return state.astype(np.float32), self.scale_factor_reward*reward, done, info

    def parse_state(self):
        # is a result of the code below. The state space has 3 dimensions. One is a scalar and 2 are vectros of length T
        input_size = 2*self.T + 1
        x = np.reshape(
            np.squeeze(
                np.concatenate((
                    np.reshape([self.env.acc[n][self.env.time+1] *
                               self.scale_factor for n in self.env.nodes], (1, 1, self.env.number_nodes)),
                    np.reshape([[(self.env.acc[n][self.env.time+1] + self.env.dacc[n][t])*self.scale_factor for n in self.env.nodes]
                                for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes)),
                    np.reshape([[sum([self.env.price[o[0], j][t]*self.scale_factor*self.scale_factor_price*(self.env.demand[o[0], j][t])*((o[1]-self.env.scenario.energy_distance[o[0], j]) >= int(not self.env.scenario.charging_stations[j]))
                                      for j in self.env.region]) for o in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes))),
                               axis=1), axis=0), (input_size, self.env.number_nodes)
        )

        return np.transpose(x)

    def close(self):
        pass

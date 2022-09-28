from cmath import inf
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from src.algos.reb_flow_solver import solveRebFlow
import gym
from gym import spaces
import wandb


class FleetEnv(gym.Env):
  """
  Custom Environment that follows gym interface. 
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['console']}

  def __init__(self, env=None, gurobi_env=None, T=10):
    super(FleetEnv, self).__init__()
    self.env = env
    self.gurobi_env = gurobi_env
    self.episode = 0
    self.episode_reward = 0
    self.episode_served_demand = 0
    self.episode_rebalancing_cost = 0
    self.T = T
    _, paxreward, _, info = self.env.pax_step(gurobi_env=self.gurobi_env)
    self.episode_served_demand += info['served_demand']
    # self.episode_reward += paxreward
    # Define action and observation space
    self.initial_state = self.parse_state().astype(np.float32)
    print(self.initial_state.shape)
    self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.env.nodes),), dtype=np.float32) # TODO potential problem with bounds and shape
    self.observation_space = spaces.Box(low=0, high=inf, shape=self.initial_state.shape, dtype=np.float32)

  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    wandb.log({"Episode": self.episode, "Reward": self.episode_reward, "ServedDemand": self.episode_served_demand, "Reb. Cost": self.episode_rebalancing_cost})
    self.episode += 1
    self.episode_reward = 0
    self.episode_served_demand = 0
    self.episode_rebalancing_cost = 0
    self.env.reset()
    _, paxreward, _, info = self.env.pax_step(gurobi_env=self.gurobi_env)
    self.episode_served_demand += info['served_demand']
    # self.episode_reward += paxreward
    return self.initial_state

  def step(self, action):
    jitter=1e-20
    concentration = F.softplus(torch.tensor(action)).reshape(-1) + jitter  # TODO decide if I need this
    m = Dirichlet(concentration)
    v_action = m.sample()
    total_acc = sum(self.env.acc[n][self.env.time+1] for n in self.env.nodes)
    desiredAcc = {self.env.nodes[i]: int(v_action[i] *total_acc) for i in range(self.env.number_nodes)}
    total_desiredAcc = sum(desiredAcc[n] for n in self.env.nodes)
    missing_cars = total_acc - total_desiredAcc
    most_likely_node = np.argmax(v_action)
    # TODO: solve problem here!!!
    if missing_cars !=0:
        desiredAcc[self.env.nodes[most_likely_node]] += missing_cars   
        total_desiredAcc = sum(desiredAcc[n] for n in self.env.nodes)
    assert total_desiredAcc == total_acc
    rebAction = solveRebFlow(env=self.env, desiredAcc=desiredAcc, gurobi_env=self.gurobi_env)
    _, reb_reward, done, info = self.env.reb_step(rebAction)
    self.episode_rebalancing_cost += info['rebalancing_cost']
    self.episode_reward += reb_reward
    reward = reb_reward
    if not done:
      _, paxreward, _, info = self.env.pax_step(gurobi_env=self.gurobi_env)
      self.episode_served_demand += info['served_demand']
      self.episode_reward += paxreward
      reward += paxreward
      state = self.parse_state()
    else:
      state = self.initial_state

    return state.astype(np.float32), reward, done, info

  def parse_state(self):
    # might have to add self.s again
    x = np.reshape(
      np.squeeze(
      np.concatenate((
        np.reshape([self.env.acc[n][self.env.time+1] for n in self.env.nodes], (1, 1, self.env.number_nodes)), 
        np.reshape([[(self.env.acc[n][self.env.time+1] + self.env.dacc[n][t]) for n in self.env.nodes] \
                      for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes)),
        np.reshape([n[0] for n in self.env.nodes], (1, 1, self.env.number_nodes)),
        np.reshape([n[1] for n in self.env.nodes], (1, 1, self.env.number_nodes)),
        # TODO: reg -> PROB, error with correspondence to nodes? Could add demand only if it is feasible, added *((o[1]-self.env.rebTime[o[0],j])>0) TODO:must add proper conversion from time to energy!!!
        np.reshape([[sum([(self.env.demand[o[0],j][t])*((o[1]-self.env.scenario.energy_distance[o[0],j])>= int(j not in self.env.scenario.charging_stations))*(self.env.price[o[0],j][t]) \
                      for j in self.env.region]) for o in self.env.nodes] for t in range(self.env.time+1, self.env.time+self.T+1)], (1, self.T, self.env.number_nodes))),
          axis=1), axis=0),(2*self.T + 3, self.env.number_nodes)
      )
    
    np.transpose(x) # TODO check if we need a transpose
    return x

  def close(self):
    pass
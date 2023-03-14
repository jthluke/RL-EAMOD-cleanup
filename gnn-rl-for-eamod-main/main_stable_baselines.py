import argparse
from locale import normalize
from statistics import mode
# from time
import gurobipy as gp
import os
import numpy as np
from src.envs.gym import FleetEnv
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
import json

import gym

from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th

from src.envs.amod_env import Scenario, AMoD


parser = argparse.ArgumentParser(description='A2C-GNN')
# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--continues', type=bool, default=False,
                    help='continues training')
parser.add_argument('--toy', type=bool, default=False,
                    help='activates toy mode for agent evaluation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=16000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--number_subproblems', type=int, default=3, metavar='N',
                    help='number of training subproblems (default: 3')
parser.add_argument('--T', type=int, default=10, metavar='N',
                    help='Time horizon for the A2C')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


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

    scenario = Scenario(spatial_nodes=spatial_nodes, charging_stations=chargers, cars_per_station_capacity=cars_per_station_capacity, number_charge_levels=number_charge_levels, charge_levels_per_charge_step=charge_levels_per_charge_step,
                        energy_distance=energy_dist, tf=tf, sd=seed, tripAttr=tripAttr, demand_ratio=1, reb_time=reb_time, total_acc=total_acc, p_energy=p_energy, time_granularity=time_granularity, operational_cost_per_timestep=operational_cost_per_timestep)
    return scenario


class CustomGraphFeature(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, edge_index: torch.Tensor = torch.Tensor()):
        self.num_nodes = observation_space.shape[0]
        self.input_features = observation_space.shape[1]
        self.output_features = features_dim
        super(CustomGraphFeature, self).__init__(
            observation_space, features_dim)

        self.edge_index = edge_index
        self.graphconv1 = GCNConv(
            self.input_features, features_dim, improved=True)
        self.graphconv2 = GCNConv(
            features_dim, features_dim, improved=True)

        # self.graphconv2 = GCNConv(features_dim, features_dim)
        # self.linear1 = nn.Sequential(nn.Linear(4, features_dim), nn.ReLU())
        # self.linear2 = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

        self.batch_size = None
        self.batch_edge_index = None

    def forward(self, observations: th.Tensor) -> th.Tensor:

        if observations.shape[0] != self.batch_size:
            # print("observation", observations.shape)
            self.batch_size = observations.shape[0]
            self.batch_edge_index = torch.cat(
                [self.edge_index + self.num_nodes*i for i in range(self.batch_size)], dim=1)

        x_batch = observations.reshape(-1, self.input_features)

        x = self.graphconv1(x_batch, self.batch_edge_index)
        x = self.graphconv2(x, self.batch_edge_index)
        x = x.reshape(self.batch_size, self.num_nodes, self.output_features)
        # x = x.reshape(self.batch_size, -1)
        # print("feature extract", x.shape)

        return x


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 2,
        last_layer_dim_vf: int = 1,
    ):
        super(CustomNetwork, self).__init__()

        self.latent_dim_pi = 190 # TODO: change for every experiment, needs to be #nodes x2 -> think of better way 192
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(
                128, 64), nn.ReLU(), nn.Linear(
                64, 32), nn.ReLU(), nn.Linear(32, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(
                128, 64), nn.ReLU(), nn.Linear(
                64, 32), nn.ReLU(), nn.Linear(32, last_layer_dim_vf)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # time_start_forward = time.time()
        batch_size = features.shape[0]
        node_size = features.shape[1]
        input_feature_size = features.shape[2]
        features_reshaped = features.reshape(-1,input_feature_size)
        action = self.policy_net(features_reshaped)
        action_reshaped = th.vstack([i.T.flatten() for i in action.split(node_size)])
        # action_reshaped = th.squeeze(action_reshaped, dim=0)

        node_sum_features = features.sum(axis=1)
        value = self.value_net(node_sum_features)
        # value = th.squeeze(value, dim=0)
        # time_end_forward = time.time()
        # print("forward time", time_end_forward - time_start_forward)
        return action_reshaped, value

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # time_start_actor = time.time()
        batch_size = features.shape[0]
        node_size = features.shape[1]
        input_feature_size = features.shape[2]
        features_reshaped = features.reshape(-1,input_feature_size)
        action = self.policy_net(features_reshaped)
        action_reshaped = th.vstack([i.T.flatten() for i in action.split(node_size)])
        # time_end_actor = time.time()
        # print("actor time", time_end_actor - time_start_actor)

        return action_reshaped

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # time_start_critic = time.time()
        node_sum_features = features.sum(axis=1)
        value = self.value_net(node_sum_features)
        # time_end_critic = time.time()
        # print("critic time", time_end_critic - time_start_critic)
        return value


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


def main():
    # set Gurobi environment mine
    # gurobi_env = gp.Env(empty=True)
    # gurobi = "Dominik"
    # gurobi_env.setParam('WLSACCESSID', '8cad5801-28d8-4e2e-909e-3a7144c12eb5')
    # gurobi_env.setParam('WLSSECRET', 'a25b880b-8262-492f-a2e5-e36d6d78cc98')
    # gurobi_env.setParam('LICENSEID', 799876)
    # gurobi_env.setParam("OutputFlag",0)
    # gurobi_env.start()

    # set Gurobi environment mine2
    gurobi_env = gp.Env(empty=True)
    gurobi = "Dominik2"
    gurobi_env.setParam('WLSACCESSID', 'ab720bb4-cdd4-40c0-a07f-79ac1ac8b13a')
    gurobi_env.setParam('WLSSECRET', '42b278c4-57fa-4ac4-a963-9edf5cad4719')
    gurobi_env.setParam('LICENSEID', 856171)
    gurobi_env.setParam("OutputFlag",0)
    gurobi_env.start()

    # set Gurobi environment Justin
    # gurobi_env = gp.Env(empty=True)
    # gurobi = "Justin"
    # gurobi_env.setParam('WLSACCESSID', '82115472-a780-40e8-9297-b9c92969b6d4')
    # gurobi_env.setParam('WLSSECRET', '0c069810-f45f-4920-a6cf-3f174425e641')
    # gurobi_env.setParam('LICENSEID', 844698)
    # gurobi_env.setParam("OutputFlag",0)
    # gurobi_env.start()

    # set Gurobi environment Karthik
    # gurobi_env = gp.Env(empty=True)
    # gurobi = "Karthik"
    # gurobi_env.setParam('WLSACCESSID', 'ad632625-ffd3-460a-92a0-6fef5415c40d')
    # gurobi_env.setParam('WLSSECRET', '60bd07d8-4295-4206-96e2-bb0a99b01c2f')
    # gurobi_env.setParam('LICENSEID', 849913)
    # gurobi_env.setParam("OutputFlag", 0)
    # gurobi_env.start()

    # set Gurobi environment Karthik2
    # gurobi_env = gp.Env(empty=True)
    # gurobi = "Karthik2"
    # gurobi_env.setParam('WLSACCESSID', 'bc0f99a5-8537-45c3-89d9-53368d17e080')
    # gurobi_env.setParam('WLSSECRET', '6dddd313-d8d4-4647-98ab-d6df872c6eaa')
    # gurobi_env.setParam('LICENSEID', 799870)
    # gurobi_env.setParam("OutputFlag",0)
    # gurobi_env.start()

    if args.toy:
        problem_folder = 'Toy'
        file_path = os.path.join(
            'data', problem_folder, 'scenario_train1x1.json')
        experiment = 'training_stable_baselines_' + problem_folder + '_' + \
            str(args.max_episodes) + '_episodes_T_' + str(args.T) + file_path
        energy_dist_path = os.path.join(
            'data', problem_folder,  'energy_distance_1x1.npy')
        scenario = create_scenario(file_path, energy_dist_path)
        env = AMoD(scenario)
        scale_factor = 0.01
        scale_price = 0.1
        # scale_factor_reward = 30/24000  # for 1x1
        scale_factor_reward = 20/6688  # for 2x1, 2x3
        # scale_factor_reward = 30/6988  # for artificial
        gym_env = FleetEnv(env=env, gurobi_env=gurobi_env, T=args.T, scale_factor_reward=scale_factor_reward,
                           scale_factor=scale_factor, price_scale_factor=scale_price, test=args.test)
    else:
        problem_folder = 'NY_5'
        file_path = os.path.join('data', problem_folder,  'NY_5_day.json')
        # problem_folder = 'SF_5_clustered'
        # file_path = os.path.join('data', problem_folder,  'SF_5_short_afternoon_test.json')
        model_name = "model_NY.json"
        experiment = 'training_stable_baselines_' + file_path + '_' + \
            str(args.max_episodes) + '_episodes_T_' + str(args.T)
        energy_dist_path = os.path.join(
            'data', problem_folder, 'energy_distance.npy')
        scenario = create_scenario(file_path, energy_dist_path)
        env = AMoD(scenario)
        # Initialize A2C-GNN
        scale_factor = 0.0001  # NY5
        scale_factor_reward = 16/1383299  # for NY5
        # scale_factor = 0.00001 # SF
        # scale_factor_reward = 16/9862821  # for SF
        scale_price = 0.1
        gym_env = FleetEnv(env=env, gurobi_env=gurobi_env, T=args.T, scale_factor_reward=scale_factor_reward,
                           scale_factor=scale_factor, price_scale_factor=scale_price, test=args.test)

    if not args.test:
        run = wandb.init(
            # Set the project where this run will be logged
            project='e-amod',
            # pass a run name
            name=experiment,
            # Track hyperparameters and run metadata
            config={
                "number_chargelevels": env.scenario.number_charge_levels,
                "number_spatial_nodes": env.scenario.spatial_nodes,
                "dataset": file_path,
                "episodes": args.max_episodes,
                "number_vehicles_per_node_init": env.G.nodes[(0, 1)]['accInit'],
                "charging_stations": list(env.scenario.charging_stations),
                "charging_station_capacities": list(env.scenario.cars_per_station_capacity),
                "scale_factor": scale_factor,
                "scale_price": scale_price,
                "time_horizon": args.T,
                "episode_length": env.tf,
                "charge_levels_per_timestep": env.scenario.charge_levels_per_charge_step,
                "licence": gurobi,
            })

        policy_kwargs = dict(
            features_extractor_class=CustomGraphFeature,
            features_extractor_kwargs=dict(
                features_dim=64, edge_index=gym_env.env.gcn_edge_idx)
        )
        # model = A2C(CustomActorCriticPolicy, gym_env,
        #             policy_kwargs=policy_kwargs, verbose=0, n_steps=20)
        if not args.continues:
            model = PPO(CustomActorCriticPolicy, gym_env,
                        policy_kwargs=policy_kwargs, verbose=0, batch_size=2, n_steps=16)
        else:
            model = PPO.load("model.json", env=gym_env)


        training_steps = int(15000000 / 10000)
        for t in range(training_steps):
            model.learn(total_timesteps=10000, callback=WandbCallback(
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ))
            model.save(f"model_"+experiment+str(t))
            wandb.save(f"model_"+experiment+str(t))
        run.finish()
    else:
        model = PPO.load(model_name, env=gym_env)
        mean, std = evaluate_policy(model=model, env=gym_env, n_eval_episodes=200)
        print(mean, std)
        wandb.finish()


if __name__ == "__main__":
    main()

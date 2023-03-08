"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

from operator import ne
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import global_mean_pool, global_max_pool
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, T=10, scale_factor=0.01, scale_price=0.1, input_size=23):
        super().__init__()
        self.env = env
        self.T = T
        self.scale_factor = scale_factor
        self.price_scale_factor = scale_price
        self.input_size = input_size
        
    def parse_obs_spatial(self):
        x = torch.cat((
            torch.tensor([self.env.acc_spatial[n][self.env.time+1]*self.scale_factor for n in self.env.nodes_spatial]).view(1, 1, self.env.number_nodes_spatial).float(), 
            torch.tensor([[(self.env.acc_spatial[n][self.env.time+1] + self.env.dacc_spatial[n][t])*self.scale_factor for n in self.env.nodes_spatial] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes_spatial).float(),
            torch.tensor([[sum([self.env.price[o,j][t]*self.scale_factor*self.price_scale_factor*(self.env.demand[o,j][t]) \
                          for j in self.env.region]) for o in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.number_nodes_spatial).float()),
              dim=1).squeeze(0).view(self.input_size, self.env.number_nodes_spatial).T
        edge_index  = self.env.gcn_edge_idx_spatial
        data = Data(x, edge_index)
        return data
    
#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 2)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        # out = F.relu(self.conv2(out, data.edge_index))
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x[:,0], x[:,1]

#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        # self.conv2 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        # out = F.relu(self.conv2(out, data.edge_index))
        x = out + data.x 
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu"), T=10, lr_a=1.e-3, lr_c=1.e-3, grad_norm_clip_a=0.5, grad_norm_clip_c=0.5, seed=10, scale_factor=0.01, scale_price=0.1):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.T = T
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.adapted_lr_a = lr_a
        self.adapted_lr_c = lr_c
        self.grad_norm_clip_a = grad_norm_clip_a
        self.grad_norm_clip_c = grad_norm_clip_c
        self.scale_factor = scale_factor
        self.scale_price = scale_price
        input_size = 2*T + 1
        self.input_size = input_size
        torch.manual_seed(seed)
        self.device = device
        
        self.actor = GNNActor(in_channels=self.input_size)
        self.critic = GNNCritic(in_channels=self.input_size)
        self.obs_parser = GNNParser(self.env, T=T, input_size=self.input_size, scale_factor=scale_factor, scale_price=scale_price)
        
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.means_concentration = []
        self.std_concentration = []
        self.to(self.device)
    
    def set_env(self, env):
        self.env = env
        self.obs_parser = GNNParser(self.env, T=self.T, input_size=self.input_size, scale_factor=self.scale_factor, scale_price=self.scale_price)
        self.means_concentration = []
        self.std_concentration = []

    # def decay_learning_rate(self, scaler_a=1, scaler_c=1):
    #     self.adapted_lr_a *= scaler_a
    #     self.adapted_lr_c *= scaler_c
    #     self.optimizers = self.configure_optimizers()
        
    def forward(self, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs_spatial().to(self.device)
        
        # actor: computes concentration parameters of a Dirichlet distribution
        a_out_concentration, a_out_is_zero = self.actor(x)
        concentration = F.softplus(a_out_concentration).reshape(-1) + jitter
        non_zero = torch.sigmoid(a_out_is_zero).reshape(-1)

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, non_zero, value
    
    def parse_obs_spatial(self):
        state = self.obs_parser.parse_obs_spatial()
        return state
    
    def select_action(self):
        concentration, non_zero, value = self.forward()
        # concentration, value = self.forward(obs)
        concentration_without_zeros = torch.tensor([], dtype=torch.float32)
        sampled_zero_bool_arr = []
        log_prob_for_zeros = 0
        for node in range(non_zero.shape[0]):
            sample = torch.bernoulli(non_zero[node])
            if sample>0:
                indices = torch.tensor([node])
                new_element = torch.index_select(concentration, 0, indices)
                concentration_without_zeros = torch.cat((concentration_without_zeros, new_element), 0)
                sampled_zero_bool_arr.append(False)
                log_prob_for_zeros += torch.log(non_zero[node])
            else:
                sampled_zero_bool_arr.append(True)
                log_prob_for_zeros += torch.log(1-non_zero[node])
        if concentration_without_zeros.shape[0]!=0:
            mean_concentration = np.mean(concentration_without_zeros.detach().numpy())
            std_concentration = np.std(concentration_without_zeros.detach().numpy())
            self.means_concentration.append(mean_concentration)
            self.std_concentration.append(std_concentration)
            m = Dirichlet(concentration_without_zeros)
            dirichlet_action = m.rsample()
            dirichlet_action_np = list(dirichlet_action.detach().numpy())
            log_prob_dirichlet = m.log_prob(dirichlet_action)
        else:
            log_prob_dirichlet = 0
        self.saved_actions.append(SavedAction(log_prob_dirichlet+log_prob_for_zeros, value))
        action_np = []
        dirichlet_idx = 0
        for node in range(non_zero.shape[0]):
            if sampled_zero_bool_arr[node]:
                action_np.append(0.)
            else:
                action_np.append(dirichlet_action_np[dirichlet_idx])
                dirichlet_idx += 1
       
        return action_np

    def select_equal_action(self):
        action = np.ones(self.env.number_nodes_spatial)/self.env.number_nodes_spatial
        return list(action)

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        # returns = [r / 4390. for r in returns] # 49000 is the maximum reward
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        log_probs = []
        values = []
        for (log_prob, value) in saved_actions:
            log_probs.append(log_prob.item())
            values.append(value.item())

        mean_value = np.mean(values)
        mean_concentration = np.mean(self.means_concentration)
        mean_std = np.mean(self.std_concentration)
        mean_log_prob = np.mean(log_probs)
        std_log_prob = np.std(log_probs)
        for (log_prob, value), R in zip(saved_actions, returns):
            # normed_log_prob = (log_prob - np.mean(log_probs)) / (np.std(log_probs) + self.eps)
            # normed_value = (value - mean_value) / (np.std(values) + self.eps)
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip_a)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip_c)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        return a_loss, v_loss, mean_value, mean_concentration, mean_std, mean_log_prob, std_log_prob
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=self.adapted_lr_a)
        # optimizers['a_optimizer'] = torch.optim.RAdam(actor_params, lr=self.adapted_lr_a)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=self.adapted_lr_c)
        # optimizers['c_optimizer'] = torch.optim.RAdam(critic_params, lr=self.adapted_lr_c)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)

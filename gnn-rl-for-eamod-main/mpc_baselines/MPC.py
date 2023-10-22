# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:09:46 2020
@author: yangk
"""
from collections import defaultdict
import numpy as np
import subprocess
from MPC_gurobi import solve_mpc
import os
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import re

class MPC:
    def __init__(self, env, gurobi_env, mpc_horizon, initial_state, noisy):
        self.env = env
        self.gurobi_env = gurobi_env
        self.mpc_horizon = mpc_horizon
        self.return_to_initial_state = initial_state
        self.noisy = noisy
        
    def MPC_exact(self):
        paxAction, rebAction = solve_mpc(env=self.env, gurobi_env=self.gurobi_env, mpc_horizon=self.mpc_horizon, return_initial_state=self.return_to_initial_state, noisy=self.noisy)
        return paxAction,rebAction


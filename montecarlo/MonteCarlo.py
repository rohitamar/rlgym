from collections import Counter
import numpy as np 
from numpy.random import choice, rand, randint
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import os 
from typing import List, Tuple

from utils.Agent import Agent 

class MonteCarlo(Agent):
    def __init__(self, q_size: Tuple, gamma: float):
        self.q = np.zeros(q_size, dtype=np.float64)
        self.cnt = np.zeros(q_size, dtype=np.int32)
        self.gamma = gamma 
        self.action_size = q_size[-1]

    def act(self, state: Tuple, eps):
        if rand() > eps:
            actions = self.q[state]
            mx = actions.max()
            return choice(np.flatnonzero(np.isclose(actions, mx)))
        return randint(self.action_size)
    
    def learn(self, states: List[Tuple], actions: List, rewards: List[float]):
        T = len(states)
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = self.gamma * G + rewards[t]
            state, action = states[t], actions[t] 
            state = tuple(int(x) for x in state)
            action = int(action)
            ind = state + (action,)
            self.cnt[ind] += 1
            self.q[ind] = self.q[ind] + (G - self.q[ind]) / self.cnt[ind]
        return G 
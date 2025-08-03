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

class OffPolicyMC(Agent):
    def __init__(self, q_size: Tuple, gamma: float):
        self.Q = np.zeros(q_size, dtype=np.float64)
        self.C = np.zeros(q_size, dtype=np.float64)
        self.gamma = gamma 
        self.action_size = q_size[-1]

    # act refers to the behavioral policy
    def act(self, state: Tuple, eps: float):
        actions = self.Q[state]
        mx = actions.max()
        best_actions = np.flatnonzero(np.isclose(actions, mx))
        
        action_choice = choice(best_actions) if rand() > eps else randint(self.action_size)

        if action_choice in best_actions:
            return action_choice, (1.0 - eps) / len(best_actions) + eps / self.action_size
        else:
            return action_choice, eps / self.action_size
            
    def learn(
        self, 
        states: List[Tuple], 
        actions: List, 
        rewards: List[float], 
        behavior_probs: List[float]
    ):
        T = len(states) 
        G = 0.0 
        W = 1.0 

        for t in range(T - 1, -1, -1):
            state = states[t]
            action = actions[t]
            prob = behavior_probs[t] 

            state = tuple(int(x) for x in state)
            action = int(action)
            ind = state + (action,)

            G = self.gamma * G + rewards[t]
            self.C[ind] += W
            self.Q[ind] = self.Q[ind] + (G - self.Q[ind]) * (W / self.C[ind])
            if self.Q[ind[:-1]].argmax() != action:
                break 
            W /= prob 
        return G 
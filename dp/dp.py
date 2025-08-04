import itertools
import numpy as np
from typing import Tuple 

class DPAgent():
    def __init__(
        self, 
        state_size: Tuple, 
        action_size: int, 
        transition_probs: dict, 
        gamma: float
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.V = np.zeros(state_size)
        self.P = np.random.randint(0, action_size, state_size)
        self.T = transition_probs
        self.gamma = gamma

    def evaluate(self) -> None:
        delta = float('inf')
        tol = 1e-4
        while delta > tol:
            cur_delta = 0
            # np.ndindex(*self.state_size) = itertools.product((range(x) for x in self.state_size)):
            copy_v = self.V.copy()
            for s in range(self.state_size):
                v = self.V[s]
                a = self.P[s]
                v_new = 0.0
                for prob, s_prime, reward, done in self.T[s][a]:
                    v_new += prob * (reward + (0 if done else self.gamma * copy_v[s_prime]))
                cur_delta = max(cur_delta, abs(v_new - v))
                self.V[s] = v_new
            
            delta = cur_delta 
        

    def improve(self) -> bool:
        policy_stable = True 
        for s in range(self.state_size):
            old = self.P[s]
            mx = float('-inf')
            new_action = -1
            for a in range(self.action_size):
                v_new = 0.0
                for prob, s_prime, reward, done in self.T[s][a]:
                    v_new += prob * (reward + (0 if done else self.gamma * self.V[s_prime]))
                if v_new > mx:
                    mx, new_action = v_new, a     
            self.P[s] = new_action   
            if new_action != old:
                policy_stable = False 
        
        return policy_stable
    
    def act(self, state):
        return self.P[state]
    

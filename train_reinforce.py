import numpy as np
import torch
import random
from collections import deque
import gymnasium as gym
import itertools 

from models.REINFORCE import REINFORCE 

max_steps = 1000
gamma = 0.9
lr = 1e-2
print_every = 1

def train_reinforce(env):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = REINFORCE(input_dim, output_dim, gamma=gamma, lr=lr)
    
    avg_reward = 10.0

    for episode in itertools.count(1):
        state, _ = env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            ep_reward += reward
            agent.add_reward(reward)

            if done: 
                break 
                
        avg_reward = 0.05 * ep_reward + 0.95 * avg_reward 

        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}: {ep_reward}")
        if avg_reward > env.spec.reward_threshold:
            print(f"Solved in {step + 1}")

        agent.learn() 

    return agent 
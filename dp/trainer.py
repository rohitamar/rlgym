from gymnasium import spaces
import numpy as np
import torch
import random

from dp.dp import DPAgent

gamma = 0.99

def seed_stuff(seed: int, env) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

def train_dp(env):
    seed_stuff(6969, env)

    # assume that state_size is an integer (not a tuple)
    # think this is fine
    # i initially thought that state_size could be a tuple, similar to what blackjack-v1 has
    # but for dp this should be fine

    agent = DPAgent(
        env.observation_space.n,
        env.action_space.n,
        env.unwrapped.P,
        gamma=gamma
    )
    
    sentinel = False
    iterations = 0
    while not sentinel:
        agent.evaluate()
        sentinel = agent.improve()
        iterations += 1
    
    print(f"Iterations taken: {iterations}")
    
    state, _ = env.reset()
    terminated = truncated = False
    total_return = 0.0

    while not terminated and not truncated:
        a = agent.act(state)      
        state, r, terminated, truncated, _ = env.step(a)
        total_return += r

    print("Trained Agent's Return:", total_return)
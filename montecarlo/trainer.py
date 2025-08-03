from collections import deque
from gymnasium import spaces
import numpy as np
import torch
import random
from itertools import count 

from montecarlo.MonteCarlo import MonteCarlo
from montecarlo.OffPolicyMC import OffPolicyMC

every = 5_000
window_size = 5_000
epsilon_start = 0.5
epsilon_end = 0.05
schedule_length = 50_000
num_episodes = 150_000
window = deque(maxlen=window_size)

def seed_stuff(seed: int, env) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

def schedule_epsilon(episode):
    return epsilon_start - (epsilon_start - epsilon_end) * min(1.0, episode / schedule_length)

def train_mc(env, writer):
    seed_stuff(6969, env)
    
    s = []
    if isinstance(env.observation_space, spaces.Tuple):
        for d in env.observation_space:
            s.append(d.n)
    else:
        s.append(env.observation_space.n)
    s.append(env.action_space.n)
    s = tuple(s)

    agent = MonteCarlo(
        q_size=s,
        gamma=1.0
    )

    tot = 0.0
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        episode_reward = 0.0
        epsilon = schedule_epsilon(episode)

        for _ in count():
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward 

            state = next_state
            if done:
                break

        G = agent.learn(states, actions, rewards)        
        window.append(G)
        tot += G

        if len(window) == window_size and episode % every == 0:
            window_mean = np.mean(window)
            mean = tot / episode
            print(f"Episode {episode} | mean {window_mean:+.3f} | overall {mean:+.3f}")
            writer.add_scalar('Overall Mean', mean, episode)
            writer.add_scalar('Last 5000 Mean', window_mean, episode)
        writer.add_scalar('Reward', episode_reward, episode)

    return agent 

def train_offpolicy_mc(env, writer):
    seed_stuff(6969, env)
    
    s = []
    if isinstance(env.observation_space, spaces.Tuple):
        for d in env.observation_space:
            s.append(d.n)
    else:
        s.append(env.observation_space.n)
    s.append(env.action_space.n)
    s = tuple(s)

    agent = OffPolicyMC(
        q_size=s,
        gamma=1.0
    )

    tot = 0.0
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        states, actions, rewards, behavior_probs = [], [], [], []
        episode_reward = 0.0
        epsilon = schedule_epsilon(episode)

        for _ in count():
            action, prob = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            behavior_probs.append(prob) 

            episode_reward += reward 

            state = next_state
            if done:
                break

        G = agent.learn(states, actions, rewards, behavior_probs)        
        window.append(G)
        tot += G

        if len(window) == window_size and episode % every == 0:
            window_mean = np.mean(window)
            mean = tot / episode
            print(f"Episode {episode} | mean {window_mean:+.3f} | overall {mean:+.3f}")
            writer.add_scalar('Overall Mean', mean, episode)
            writer.add_scalar('Last 5000 Mean', window_mean, episode)
        writer.add_scalar('Reward', episode_reward, episode)

    return agent 

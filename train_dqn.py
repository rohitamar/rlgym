import numpy as np
import torch
import random
from collections import deque
import gymnasium as gym

from models.DQNAgent import DQNAgent

num_episodes = 1000
max_steps = 200
epsilon_start = 1.0
epsilon_end = 0.2
epsilon_decay_rate = 0.99
gamma = 0.9
lr = 1e-2
buffer = deque(maxlen=10000)
batch_size = 128
print_every = 1

def train_dqn(env):
    seed = 6969

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DQNAgent(input_dim, output_dim, gamma=gamma, lr=lr, device=device)

    for episode in range(num_episodes):
        state, _ = env.reset()
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))

        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated

            buffer.append((state, action, reward, next_state, done))
            episode_reward += reward 

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                agent.learn(batch)

            state = next_state

            if done:
                break

        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}: {episode_reward}")

    agent.save_weights()
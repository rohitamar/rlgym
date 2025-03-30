import numpy as np
import torch
import random
from itertools import count

from dqn.dqn import DQNAgent
from utils.replaybuffer import ReplayBuffer

num_episodes = 600
max_steps = 500
epsilon_start = 0.9
epsilon_end = 0.05
epsilon_decay_rate = 0.99
gamma = 0.99
lr = 1e-3
batch_size = 256
print_every = 1

def train_dqn(env, device, writer):
    seed = 1200

    buffer = ReplayBuffer(10000, 150, 0.1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DQNAgent(input_dim, 
                     output_dim, 
                     gamma=gamma, 
                     lr=lr, 
                     device=device)

    cnt_loss_step = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, device=device)
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay_rate ** episode))
        episode_reward = 0.0

        for _ in count():
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            action = torch.tensor(action, device=device)
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, device=device)
            done = torch.tensor([done], device=device)

            buffer.push(state, action, reward, next_state, done, episode)
            episode_reward += reward[0].item()

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.learn(batch)
                writer.add_scalar('Loss', loss, cnt_loss_step)
                cnt_loss_step += 1

            state = next_state
            if done:
                break

        print(f"Episode {episode + 1}: {episode_reward}")
        writer.add_scalar('Reward', episode_reward, episode + 1)
    
    return agent 

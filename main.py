from torch.utils.tensorboard import SummaryWriter
import torch
import gymnasium as gym
from datetime import datetime

from dqn.train_dqn import train_dqn
from reinforce.train_reinforce import train_reinforce 

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./runs/dqn-{timestamp}"
    writer = SummaryWriter(path)
    env = gym.make("CartPole-v1", render_mode='human')
    print(f"Target reward: {env.spec.reward_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training models on {device}.")

    agent = train_dqn(env, device, writer)
    agent.save_weights(path)


from datetime import datetime
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch

from montecarlo.trainer import train_mc

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"./runs/montecarlo-{timestamp}"
writer = SummaryWriter(path)
env = gym.make("Blackjack-v1", sab=False)
# print(f"Target reward: {env.spec.reward_threshold}")

agent = train_mc(env, writer)
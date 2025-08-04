from datetime import datetime
import gymnasium as gym
import torch

from dp.trainer import train_dp

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"./runs/dp-{timestamp}"
env = gym.make(
    "FrozenLake-v1", 
    map_name="8x8",
    is_slippery=False,
    render_mode="human"
)
# env = gym.make(
#     "Taxi-v3",
#     render_mode="human"
# )
print(f"Target reward: {env.spec.reward_threshold}")

agent = train_dp(env)
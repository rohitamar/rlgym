import gymnasium as gym
import time 
import torch 

from models.DQNAgent import DQNAgent

test_episodes = 100
episode_rewards = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

env = gym.make("CartPole-v1", render_mode='human')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent = DQNAgent(input_dim, output_dim, device=device, lr = 1e-3)
agent.load_weights('./checkpoints/dqn_20250130_150638.pth')

for episode in range(test_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated 
        episode_reward += reward
        state = next_state

    episode_rewards.append(episode_reward)

env = gym.make('CartPole-v1', render_mode='human') 
state, _ = env.reset() 
done = False

while not done:
    action = agent.act(state)  
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  
    state = next_state
    time.sleep(0.1)  

env.close()
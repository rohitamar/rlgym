import itertools 
import numpy as np 

from reinforce.reinforce import REINFORCE
from reinforce.baseline import Baseline

max_steps = 1000
gamma = 0.99
lr = 1e-3

def train_reinforce(env, device, writer):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = REINFORCE(input_dim, output_dim, gamma=gamma, device=device, lr=lr)
    
    reward_history = []
    avg_reward = 10.0

    for episode in itertools.count(1):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            episode_reward += reward
            
            agent.add_reward(reward)
            state = next_state
            if done: 
                break 
        
        agent.learn()
        print(f"Episode {episode}: {episode_reward}")
        writer.add_scalar('Episode Reward', episode_reward, episode)
        
        reward_history.append(episode_reward)
        if len(reward_history) > 100:
            reward_history.pop(0)

        avg_reward = np.mean(reward_history)
        if len(reward_history) == 100 and avg_reward > env.spec.reward_threshold:
            print(f"Solved in {episode}")

    return agent 
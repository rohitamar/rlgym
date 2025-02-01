from train_dqn import train_dqn
from train_reinforce import train_reinforce 

import gymnasium as gym

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode='human')
    print(f"Target reward: {env.spec.reward_threshold}")

    agent = train_dqn(env) 

import itertools 
from reinforce.reinforce import REINFORCE

max_steps = 1000
gamma = 0.9
lr = 1e-4
print_every = 1

def train_reinforce(env, device, writer):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = REINFORCE(input_dim, output_dim, gamma=gamma, device=device, lr=lr)
    
    avg_reward = 10.0

    for episode in itertools.count(1):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            
            action = agent.act(state)
            
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 
            episode_reward += reward
            agent.add_reward(reward)

            if done: break 
                
        avg_reward = 0.05 * episode_reward + 0.95 * avg_reward 

        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}: {episode_reward}")
        
        writer.add_scalar('Episode Reward', episode_reward, episode + 1)

        if avg_reward > env.spec.reward_threshold:
            print(f"Solved in {step + 1}")

        agent.learn() 

    return agent 
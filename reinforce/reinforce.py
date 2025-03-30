import torch 
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn 
from torch.distributions import Categorical 

from utils.Agent import Agent 

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim = -1)

class REINFORCE(Agent):
    def __init__(self, state_size, action_size, device, gamma=0.99, lr=1e-2):
        super(REINFORCE, self).__init__()
        self.device = device 
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.gamma = gamma 
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.log_probs = []
        self.rewards = []
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def add_reward(self, r):
        self.rewards = [r] + self.rewards 

    def learn(self):
        R = 0
        dis_rewards = [] # discounted_rewards 
        for reward in self.rewards:
            R = reward + self.gamma * R 
            dis_rewards = [R] + dis_rewards
        
        dis_rewards = torch.tensor(dis_rewards)
        dis_rewards = (dis_rewards - dis_rewards.mean()) / dis_rewards.std()
        loss = []
        for g_t, log_prob in zip(dis_rewards, self.log_probs):
            loss.append(g_t * -log_prob)    
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.log_probs = []




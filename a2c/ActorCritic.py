import os
import torch 
import torch.nn.functional as F
import torch.optim as optim 
import torch.nn as nn 
from torch.distributions import Categorical 

from utils.Agent import Agent 

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size=24):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim = -1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size, layer_size=24):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic(Agent):
    def __init__(self, state_size, action_size, device, gamma=0.99, lr=1e-2):
        super(ActorCritic, self).__init__()
        self.device = device 
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.value = ValueNetwork(state_size).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma 

        self.log_probs = []
        self.rewards = []
        self.states = []
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.states.append(state)
        return action.item()

    def add_reward(self, reward):
        self.rewards.append(reward)
    
    def learn(self, next_states):
        states = torch.stack(self.states)
        next_states = torch.stack(next_states)

        v_s = self.value(states)
        v_s_next = self.value(next_states)

        policy_loss = torch.stack(loss).sum()
        value_loss = F.mse_loss(v_t.squeeze(), dis_rewards)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward() 
        self.value_optimizer.step()

        self.rewards = []
        self.log_probs = []
        self.states = []
    
    def save_weights(self, path):
        filename = os.path.join(path, 'weights.pth')
        6
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filename)
    
    def load_weights(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
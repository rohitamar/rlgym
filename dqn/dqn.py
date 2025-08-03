import numpy as np 
from numpy.random import rand, randint 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

from utils.Agent import Agent
from utils.replaybuffer import Transition

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_size=128):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.layer3 = nn.Linear(layer_size, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent(Agent):
    def __init__(self, state_size: int, action_size: int, *, gamma, lr, device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma 
        self.device = device 

        self.local: nn.Module = QNetwork(state_size, action_size).to(self.device)
        self.target: nn.Module = QNetwork(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.local.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
    
    def save_weights(self, path: str) -> None:
        filename = os.path.join(path, 'weights.pth')
        
        torch.save({
            'local_state_dict': self.local.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filename)
    
    def load_weights(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])

        assert self.state_size == checkpoint['state_size'], \
               "State size mismatch between model and checkpoint"
        
        assert self.action_size == checkpoint['action_size'], \
               "Action size mismatch between model and checkpoint"
        
        print(f"Model weights loaded from {filename}")

    def act(self, state: torch.Tensor, eps=0.):
        state = state.unsqueeze(0)
        self.local.eval()
        with torch.no_grad():
            action_values = self.local(state)

        if rand() > eps:
            return action_values.argmax(dim=1).item()
        return randint(self.action_size)

    def learn(self, experiences=None):
        self.local.train()

        batch = Transition(*zip(*experiences))
        
        states = torch.vstack(batch.state).float()
        actions = torch.vstack(batch.action).long()
        rewards = torch.vstack(batch.reward).float()
        next_states = torch.vstack(batch.next_state).float()
        dones = torch.vstack(batch.done).long()

        Q_targets_next = self.target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_preds = self.local(states).gather(1, actions)

        loss = F.mse_loss(Q_preds, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.local.parameters(), 100)
        self.optimizer.step() 
        self.scheduler.step()

        for target, local in zip(self.target.parameters(), self.local.parameters()):
            target.data.copy_(0.005 * local.data + (1.0 - 0.005) * target.data)
        



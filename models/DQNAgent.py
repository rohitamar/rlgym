import os 
from datetime import datetime
import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma, lr, device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma 
        
        self.device = device 

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)
    
    def save_weights(self):
        os.makedirs("checkpoints", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./checkpoints/dqn_{timestamp}.pth"
        
        torch.save({
            'local_state_dict': self.qnetwork_local.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size
        }, filename)
    
    def load_weights(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])

        assert self.state_size == checkpoint['state_size'], \
               "State size mismatch between model and checkpoint"
        
        assert self.action_size == checkpoint['action_size'], \
               "Action size mismatch between model and checkpoint"
        
        print(f"Model weights loaded from {filename}")

    def act(self, state, eps=0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        return action_values.argmax(dim=1).item() if np.random.random() > eps else np.random.randint(self.action_size)

    def learn(self, experiences=None):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_preds = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_preds, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, alpha=1e-3)

    def soft_update(self, local_model, target_model, alpha):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(alpha * local_param.data + (1.0 - alpha) * target_param.data)
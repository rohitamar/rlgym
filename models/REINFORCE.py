import torch 
import torch.nn.fucntional as F
from torch.distributions import Categorical 

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
    def __init__(self, state_size, action_size):
        super(REINFORCE, self).__init__()
        self.policy = PolicyNetwork()

    def save_weights(self):
        pass 

    def load_weights(self, filename):
        pass

    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()
    

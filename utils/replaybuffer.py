from collections import namedtuple, deque
from random import sample

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, thresh_episode, initial_percent):
        self.init_memory = []
        self.memory = deque([], maxlen=capacity)
        self.thresh_episode = thresh_episode
        self.initial_percent = initial_percent
    
    def push(self, state, action, reward, next_state, done, episode):
        t = Transition(state, action, reward, next_state, done)
        if episode < self.thresh_episode:
            self.init_memory.append(t)
        self.memory.append(t)

    def sample(self, batch_size):
        from_initial = int(self.initial_percent * batch_size)
        return sample(self.memory, batch_size - from_initial) + sample(self.init_memory, from_initial)

    def __len__(self):
        return len(self.memory)
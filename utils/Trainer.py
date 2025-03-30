from abc import ABC, abstractmethod 
import numpy as np
import torch
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Trainer(ABC):
    def __init__(self, **kwargs):
        # check if variables env and device are passed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def before_train(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed) 

        class_name = self.__class__.__name__ 
        position = class_name.find("Trainer")
        agent_type = class_name[:position]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = self.env.spec.id 
        self.path = f"./runs/{agent_type}-{timestamp}-{env_name}"

        self.writer = SummaryWriter(self.path)

    @abstractmethod
    def train(self):
        pass 
    
    def train_loop(self):
        self.before_train()
        agent = self.train() 
        agent.save_weights(self.path)
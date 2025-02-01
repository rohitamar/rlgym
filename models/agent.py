from abc import ABC, abstractmethod 

class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass 

    @abstractmethod 
    def learn(self):
        pass 
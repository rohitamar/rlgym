from abc import ABC, abstractmethod 

class Agent(ABC):
    @abstractmethod
    def act(self):
        pass 

    @abstractmethod 
    def learn(self):
        pass 
    
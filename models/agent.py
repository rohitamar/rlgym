from abc import ABC, abstractmethod 

class Agent(ABC):

    @abstractmethod 
    def act():
        pass 

    @abstractmethod 
    def learn():
        pass 

    
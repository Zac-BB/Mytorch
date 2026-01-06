from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    def __init__(self,model,loss,lr):
        super().__init__()

    @abstractmethod
    def backward():
        pass

    @abstractmethod
    def step():
        pass
    

class SGD(Optimizer):
    def __init__(self, model, loss,lr):
        super().__init__(model, loss,lr)
    
    def backward():
        pass
    
    def step():
        pass
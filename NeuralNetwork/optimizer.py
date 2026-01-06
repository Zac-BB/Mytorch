from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    def __init__(self,model,loss):
        super().__init__()
    
    def backward():
        pass
    

class SGD(Optimizer):
    def __init__(self, model, loss,lr):
        super().__init__(model, loss)
    
    def backward():
        pass
from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    def __init__(self,model,loss,):
        super().__init__()
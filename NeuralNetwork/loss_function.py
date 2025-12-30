from abc import ABC, abstractmethod
import numpy as np

class LossFunction(ABC):
    @abstractmethod
    def loss(self,y,y_hat):
        pass
    @abstractmethod
    def gradient(self,y,y_hat):
        pass

class CrossEntropyLoss(LossFunction):

    def loss(self,y,y_hat):
        log_y_hat = np.log(y_hat).T
        sum = y @ log_y_hat
        return sum
    
    def gradient(self, y, y_hat):# in the case of the softmax being used
        return 1
    
class MeanSquaredError(LossFunction):

    def loss(self,y,y_hat):
        return 1/(y.size) * (y -y_hat) @ (y -y_hat).T
    
    def gradient(self, y, y_hat):
        return 2/(y.size) * (y -y_hat)
    


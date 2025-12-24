from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    # @staticmethod
    @abstractmethod
    def __call__(self,zi):
        # called in the forward step 
        pass

    # @staticmethod
    @abstractmethod
    def partial_derivative(self,zi):
        # calculates the partial derivative d a^i_j / d z^i_j
        pass

class ReLU(ActivationFunction):
    # @staticmethod
    def __call__(self,zi):
        return np.maximum(0, zi)
    
    # @staticmethod
    def partial_derivative(self,zi):
        return (zi > 0).astype(float)
    
class Linear(ActivationFunction):
    def __call__(self,zi):
        return zi
    def partial_derivative(self,zi):
        return 1
 
class Softmax(ActivationFunction):
    def __call__(self, zi):
        shifted = zi - np.max(zi, axis=0, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def partial_derivative(self, zi, y_true):
        """
        zi: logits (batch_size, num_classes)
        y_true: one-hot labels
        """
        y_pred = self(zi)
        return y_pred - y_true
    


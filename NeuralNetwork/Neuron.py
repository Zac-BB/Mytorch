import numpy as np
class Neuron:
    def __init__(self,bias,weights):
        self.weights = weights
        self.bias = bias 

    def forward(self,input):
        assert weights.size == input.size, "Provided inputs and expected inputs are different sizes"
        z = weights@input.reshape([weights.size,1]) + self.bias
        return z
    def linear_form(self):
        return np.append(self.weights,self.bias)
if __name__ == "__main__":  
    weights = np.array([1,2,3])
    bias = 2
    my_neuron = Neuron(bias,weights)
    my_neuron.forward(np.array([1,1,1]))
    print(my_neuron.linear_form())

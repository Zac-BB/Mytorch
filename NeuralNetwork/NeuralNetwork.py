import numpy as np
class NeuralNetwork:
    def __init__(self,input_size,hidden_layer_size,output_size,activation_func):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.layers = []

        sizes = [input_size] + hidden_layer_size + [output_size]
        print(sizes)
        for i in range(len(sizes)-1):
            weights_i = np.ones([sizes[i+1],sizes[i]]) #0.1*np.random.randn(sizes[i+1],sizes[i])
            bias_i = np.zeros([sizes[i+1]])
            self.layers.append((weights_i,bias_i))
    
    def forward(self,input):
        layer_output = [input]
        for weights,bias in self.layers:
            linear_output = weights@layer_output[-1] + bias
            layer_output.append(linear_output)
        print(layer_output)
        return layer_output[-1]


if __name__ == "__main__":
    my_network = NeuralNetwork(3,[5,5],3,lambda x: max(0,x))
    output = my_network.forward(np.array([1,1,1]))
    print(output)
import numpy as np
class NeuralNetwork:
    def __init__(self,input_size,hidden_layer_size,output_size,activation_func):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        assert len(activation_func) == len(hidden_layer_size)+1,"Activation function size mismatch"
        self.activation_func = activation_func
        self.layers = []

        sizes = [input_size] + hidden_layer_size + [output_size]
        print(sizes)
        for i in range(len(sizes)-1):
            weights_i = 0.1*np.random.randn(sizes[i+1],sizes[i])
            bias_i = np.zeros([sizes[i+1]])
            self.layers.append((weights_i,bias_i))
    
    def forward(self,input):
        layer_output = input
        layer_outputs = [layer_output]
        for i,layer in enumerate(self.layers):
            weights,bias = layer
            linear_output = layer_output@weights.T + bias
            activation_func = self.activation_func[i]
            layer_output = activation_func(linear_output)
            layer_outputs.append(layer_output)
        # print(layer_outputs)
        return layer_output[-1]
    
    def __call__(self, *args, **kwds):
        return self.forward(*args,**kwds)
    
    def set_weights(self,weights):
        self.layers = weights

    
if __name__ == "__main__":
    from NeuralNetwork.activation import ReLU, Linear
    relu = ReLU()
    linear = Linear()
    my_network = NeuralNetwork(4,[5],3,[relu,linear])
    weights = [(np.array([[-0.46684736, -0.36128312, -0.257765  ,  0.31546897],
       [ 0.29316062, -0.22174752, -0.01804119,  0.31978035],
       [ 0.49706656,  0.19844109,  0.06754643,  0.33524317],
       [-0.29440117,  0.09317201, -0.38765275, -0.34654307],
       [-0.25829178,  0.22623652,  0.2010802 , -0.29617625]]), 
       np.array([ 0.15105355,  0.274486  , -0.06310868,  0.01909077,  0.11585236])), 
       (np.array([[ 0.3101883 ,  0.48009706, -0.38531178, -0.18323487,  0.19650495],
       [ 0.4142747 ,  0.43510365,  0.44117838,  0.09950727, -0.43479133],
       [ 0.04599625, -0.31280267, -0.46597707,  0.44424623,  0.38017988]]), 
       np.array([-0.49876398,  0.09358603, -0.08423001]))]

    my_network.set_weights(weights)
    output = my_network.forward(np.array([1,-2,0.5,3]))
    print(output)
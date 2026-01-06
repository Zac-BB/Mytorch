# from NeuralNetwork.NeuralNetwork import NeuralNetwork
# from NeuralNetwork.activation import Softmax, Linear,ReLU
from NeuralNetwork import NeuralNetwork, Softmax, Linear, one_hot
from typing import Tuple, List

import numpy as np

training_data = [
    ([48.8575,2.3514],0),
    ([48.8584,2.2945],0),
    ([48.8530,2.3499],0),
    ([48.8606,2.3376],0),
    ([48.8606,2.3522],0),

    ([40.4167,-3.7033],1),
    ([40.4153,-3.6835],1),
    ([40.4180,-3.7143],1),
    ([40.4138,-3.6921],1),
    ([40.4169,-3.7033],1),

    ([52.5200,13.4050],2),
    ([52.5163,13.3777],2),
    ([52.5169,13.4019],2),
    ([52.5074,13.3904],2),
    ([52.5251,13.3694],2),

    ]




labels = ("Paris", "Madrid", "Berlin")

if False:
    import pickle
    filename = 'cities_data.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(training_data, file)

# print(training_data)

grab_first = lambda x: (np.array([x[0][1]]),x[1])

single_dof_data = list(map(grab_first,training_data))

x = [x for x,y in single_dof_data]
y = [y for x,y in single_dof_data]
print(x)


print(single_dof_data)
softmax = Softmax()
linear = Linear()
net = NeuralNetwork(1,[],3,[softmax])
print(net.layers)


layers = {
    "W": [np.array([1.0,0.10,-1.0]).reshape(-1, 1)],
    "b": [np.zeros([3])]
}
print(layers)
net.set_weights(layers)

out = net(np.array([[2.3514]]))

print(out)



def manual_gradient(x: np.ndarray,y_hat: np.ndarray,y: np.ndarray) -> Tuple[np.ndarray]:
    input_size, output_size = 1, 3
    x = x.reshape(1, input_size)        # (1,1)
    y_hat = y_hat.reshape(output_size, 1)
    y = y.reshape(output_size, 1)

    error = y_hat - y          # (3,1)
    dw = error @ x.T           # (3,1)
    db = error.flatten()       # (3,)

    return dw, db
    # x = np.reshape(x,(input_size,))
    # y_hat = np.reshape(y_hat,(output_size,))
    # y = np.reshape(y,(output_size,))
    # dw = np.zeros((output_size,input_size))
    # db = np.zeros(output_size)
    # for row_i in range(output_size):
    #     error = (y_hat[row_i] - y[row_i])
    #     for col_j in range(input_size):
    #         dw[row_i][col_j] = x[col_j] * error
    #     db[row_i] = error

    # return dw,db




y = one_hot(np.array([0]),3)
print(manual_gradient(np.array([[2.3514]]),out,y))
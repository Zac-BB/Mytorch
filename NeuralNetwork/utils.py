import numpy as np

def one_hot(labels: np.ndarray,num_classes: int) -> np.ndarray:
    output = np.zeros((labels.size,num_classes))
    for i,label in enumerate(labels):
        output[i,label] = 1
    return output
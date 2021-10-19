import numpy as np

def create_onehot(num_features: np.number, index: int):
    onehot = np.zeros(num_features)
    onehot[index] = 1
    return onehot
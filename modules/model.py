from layers import Dense
from activations import sigmoid
from tqdm import tqdm
import numpy as np

class Model:
    '''
    It contains a list of layers and contains functions for training(forward and backward propogation) and function 
    for inference
    Model([
        Dense(),
        Dense(),
    ])
    '''
    def __init__(self, *layers):
        self.layers = layers

    def train(self, inputs, targets, verbose=False):
        '''Gets the input and then  does forward prop and then backward'''
        print(inputs.shape[0],targets.shape[0])
        assert inputs.shape[0] == targets.shape[0]
        temp = inputs

        for layer in self.layers:
            print(type(layer))
            temp = layer.call(temp)
        print(temp)

if __name__ == "__main__":
    model = Model([
        Dense(1,'relu',output_dim=4),
        Dense(4,'relu',output_dim=5),
        Dense(5,'relu',output_dim=1),
    ])

    model.train(inputs=np.array([1,2,3,4,5,6,7,8,9,10]), targets=np.array([0,1,0,1,0,1,0,1,0,1]))


                





    
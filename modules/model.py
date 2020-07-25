from layers import Dense
from activations import sigmoid
from tqdm import tqdm

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

    def fit(inputs, targets, verbose=False):
        '''Gets the input and then  does forward prop and then backward'''
        assert inputs.shape[0] == targets.shape[0]

                





    
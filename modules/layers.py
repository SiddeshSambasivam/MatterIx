import numpy as np
import activations as ac
import os

class Dense:

    def __init__(self, number_neurons, activation, prevLayerdim=2):
        self.wl = self.initialzeWeights(number_neurons)
        self.b = np.random.randn(number_neurons,1)

    def initialzeWeights(self, number_neurons, prevLayerdim=2):
        '''
        Method: He Initialization

        Args: number of neurons
        returns: initialized weights
        '''

        return np.random.randn(number_neurons,1) * np.sqrt(2/prevLayerdim) 
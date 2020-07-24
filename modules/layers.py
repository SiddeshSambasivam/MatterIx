import numpy as np
import activations as ac
from registry import ACTIVATION_REGISTRY

class Dense:

    def __init__(self, number_neurons, activation, prevLayerdim=2):
        self.wl = self.initialzeWeights(number_neurons)
        self.b = np.random.randn(number_neurons,1)
        self.units = number_neurons
        self.activation = ACTIVATION_REGISTRY[activation]
    
    def getLen(self):
        return self.units

    def initialzeWeights(self, number_neurons, prevLayerdim=2):
        return np.random.randn(number_neurons,1) * np.sqrt(2/prevLayerdim) 
    
    def forward(self, x):
        try:
            assert x.shape[0] == self.wl.shape[0]
        except:
            raise Exception 

        z1 = self.wl.dot(x.T) + self.b

        out = self.activation(z1)
    








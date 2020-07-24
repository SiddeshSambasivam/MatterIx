import numpy as np
import activations as ac
from registry import ACTIVATION_REGISTRY

np.random.seed(0)

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
    
    def call(self, x):
        try:
            assert x.shape[0] == self.wl.shape[0]
        except:
            raise Exception 
        
        z1 = np.dot(x, self.wl) + self.b
        print(z1)
        print(self.activation.__name__)
        out = self.activation()(z1.flatten())
        print(ACTIVATION_REGISTRY['add'](1,2))

if __name__ == "__main__":
    l1 = Dense(number_neurons=3, activation='sigmoid')
    x = np.array([2,3,10],)
    print(l1.call(x))
    








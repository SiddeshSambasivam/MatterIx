import numpy as np
import activations as ac
from registry import ACTIVATION_REGISTRY

np.random.seed(0)

class Dense:

    def __init__(self, number_neurons,activation, prevLayerdim=2, output_dim=None):
        
        self.input_dim = number_neurons
        
        if output_dim != None:
            self.output_dim = output_dim
        else:
            self.output_dim = number_neurons
        
        self.wl = self.initialzeWeights(number_neurons)
        self.b = np.random.randn(number_neurons,1)

        self.out = None

        if activation not in ACTIVATION_REGISTRY:
            err_msg = f"Activation function {activation} not registered"
            raise ValueError(err_msg)

        self.activation = ACTIVATION_REGISTRY[activation]
        
    
    def initialzeWeights(self, prevLayerdim=2):
        print((np.random.randn(self.input_dim,self.output_dim) * np.sqrt(2/prevLayerdim)).shape)
        return np.random.randn(self.input_dim,self.output_dim) * np.sqrt(2/prevLayerdim) 
    
    def call(self, x):
        try:
            assert x.shape[0] == self.wl.shape[0]
            x = x.reshape((x.shape[0],1))
        except:
            raise Exception 
        
        z1 = np.dot(x.T, self.wl) 
        
        print("shalpe:",self.b.shape)
        print("shalpe:",z1.shape)
        self.out = self.activation(z1)
        print(self.out.shape)
        return self.out 
    
    def get_layer(self):
        return self.out


if __name__ == "__main__":
    l1 = Dense(number_neurons=3, activation='sigmoid')
    x = np.array([2,3,10],)
    print(l1.call(x))
    # print(l1.get_layer())
    








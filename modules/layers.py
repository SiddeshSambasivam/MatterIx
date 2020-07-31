import numpy as np
import activations as ac
from registry import ACTIVATION_REGISTRY

np.random.seed(0)

class Dense:

    def __init__(self, input_dim,activation, prevLayerdim=2, output_dim=None):
        
        self.input_dim = input_dim
        
        if output_dim != None:
            self.output_dim = output_dim
        else:
            self.output_dim = input_dim
        
        self.wl = self.initialzeWeights(input_dim)
        self.b = np.random.randn(self.output_dim)

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
        
        z1 = x.T @ self.wl + self.b
        
        print("shape:",self.b.shape)
        print("shape:",z1.shape)
        self.out = self.activation(z1)
        print(self.out.shape)
        return self.out
    
    def get_layer(self):
        return self.out


if __name__ == "__main__":
    l1 = Dense(input_dim=3, output_dim =10,activation='sigmoid')
    x = np.array([2,3,10],)
    l1.call(x)
    print(l1.get_layer())
    








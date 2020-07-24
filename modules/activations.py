import numpy as np
import typing
import math
from registry import register_activation

@register_activation
def sigmoid(x):
    return 1/(1+np.exp(-x))

@register_activation
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

@register_activation
def relu(x):
    return np.maximum(x,0)

@register_activation
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

@register_activation
def lrelu(x, slope=0.01):
    y1 = (x>=0) * x
    y2 = (x<0)*slope*x
    return y1+y2

if __name__ == "__main__":
    x = np.array([-1,2,3,10])
    print(tanh(x))
    print(sigmoid(x))
    print(relu(x))
    print(softmax(x))
    print(lrelu(x))



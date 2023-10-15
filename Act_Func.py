import numpy as np
from Activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        
        def derivative_sig(x):
            # source: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
            return sigmoid(x) * (1 - sigmoid(x))
        
        super().__init__(sigmoid, derivative_sig)
    
    
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            if x > 0:
                return x
            else:
                return 0
            
        def derivative_relu(x):
            if x > 0:
                return 1
            else:
                return 0
        super().__init___(relu, derivative_relu)
    
    
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def derivative_tanh(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, derivative_tanh)


class Softmax:
    # TODO: derivative of softmax in same shape of activation and z^L
    def forward(self, input):
        self.input = input
        return self.forward_func(self.input)
    
    def backward(self, grad):
        raise NotImplementedError()
    

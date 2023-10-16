import numpy as np
from Activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return np.where(x >= 0, 1 / (1+np.exp(-x)), np.exp(x) / (1+np.exp(x)))
        
        def derivative_sig(x):
            # source: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
            s = sigmoid(x)
            return s * (1 - s)
        
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
    def forward(self, input):
        top = np.exp(input)
        self.output = top / np.sum(top)
        return self.output # should be in the shape given, for MNIST it should be 10
    
    # seeing indepedent code's video made this make so much more sense, probably best video for derivative
    def backward(self, grad, learning_rate):
        n = np.size(self.output)
        return(np.dot((np.identity(n) - self.output.T) * self.output, grad)) 

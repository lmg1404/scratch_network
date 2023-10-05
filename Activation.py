from Layer import Layer

class Activation(Layer):
    def __init__(self, forward_func, backwards_func):
        self.forward_func = forward_func
        self.backwards_func = backwards_func
        
    def forward(self, input):
        self.input = input
        return self.forward_func(self.input)
    
    def backward(self, grad):
        return grad * self.backward_func(self.input) # this should be a hadamard product, scalar by vector/element wise for vectors
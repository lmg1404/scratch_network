from Layer import Layer

class Activation(Layer):
    def __init__(self, forward_func, backwards_func):
        # these are gonna be functions, makes it easier
        self.forward_func = forward_func
        self.backward_func = backwards_func
        
    def forward(self, input):
        # input is saved for back prop
        # see my onenote on iPad
        self.input = input
        return self.forward_func(self.input)
    
    def backward(self, grad):
        # this should be a hadamard product, scalar by vector/element wise for vectors
        return grad * self.backward_func(self.input) 
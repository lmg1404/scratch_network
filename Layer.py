import numpy as np

class Layer:
    def __init__(self, input_shape, output_shape, initializiation="random"):
        if initializiation.lower() == "random":
            self.weights = np.random.randn(output_shape, input_shape)
            self.bias = np.random.randn(output_shape, 1)
            
        elif initializiation.lower() == "zero":
            self.weights = 0
            self.bias = 0
            
        elif initializiation.lower() == "he-et-al":
            # found in this Medium article:
            # https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            self.weights = np.random.randn(output_shape, input_shape) * np.sqrt(2/input_shape)
            self.bias = np.random.randn(output_shape, 1) * np.sqrt(2/input_shape)
            
        else:
            raise TypeError("Initialization not random, zero, or he-et-al")
    
    def back_prop(self):
        # TODO: look for resources to create back propagration
        pass
    
    def get_params(self, weights=True, bias=True):
        if weights and bias:
            return self.weights, self.bias
        elif weights:
            return self.weights
        elif bias:
            return self.bias
        else:
            raise TypeError("Pick a parameter to view")
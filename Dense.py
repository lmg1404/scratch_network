import numpy as np
from Layer import Layer

np.random.seed(0)
class Dense(Layer):
    """
    Class for a feedforward NN using dense architecture. This is the generic architecture.
    
    ...
    
    Attributes
    -----------
    weights : np.array
        Weights of the particular layer defined for use
    bias : np.array
        Bias (constant term) of particular layer defined for use
        
    Methods
    -----------
    forwards(input)
        Forward propagation to solve for a particular example given
    backward()
        Backward propagation to adjust weights and bias to converge the layer
    """
    
    
    def __init__(self, output_shape: int, input_shape: int, initializiation="random"):
        """
        Constructs weights and bias depending on initialization for the layer.
        
        Parameters
        -----------
        output_shape : int
            Shape of the input for the next layer
        input_shape : int
            Shape of input for this layer
        initialization : str
            Type of random initialization for weights and bias
        """
        
        # adjusted ordering that makes more sense with matrices
        # i.e. easier to read where it's m x n
        
        if initializiation.lower() == "random":
            self.weights = np.random.randn(output_shape, input_shape) * 0.01
            self.bias = np.random.randn(output_shape, 1) * 0.01
            
        elif initializiation.lower() == "zero":
            self.weights = np.zeros((output_shape, input_shape))
            self.bias = np.zeros((output_shape, 1))
            
        elif initializiation.lower() == "he-et-al":
            # found in this Medium article:
            # https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e
            self.weights = np.random.randn(output_shape, input_shape) * np.sqrt(2/input_shape)
            self.bias = np.random.randn(output_shape, 1) * np.sqrt(2/input_shape)
            
        else:
            raise TypeError("Initialization not random, zero, or he-et-al")
       
        
    def forward(self, input):
        """
        Matrix multiplies weights and the example or other output then addes bias for further processing
        
        Parameters
        -----------
        input : np.array
            Data from the previous layer for processing, must be a column vector
        
        Returns
        -----------
        z : np.array
            Processed data that is matrix multiplied by particular weight with bias added
        """
        self.input = input
        z = np.dot(self.weights, input) + self.bias
        return z
        
    
    def backward(self, grad, learning_rate=0.5):
        """
        Backwards propagation for a dense layer
        
        Parameters
        -----------
        grad : np.array
            Gradients calculated from layer ahead, a vector
        learning_rate : float
            Step size to adjust the weights and bias
            
        Returns
        -----------
        np.array
            Array to be fed into previous layers to do backpropagation, a vector
        """
        weights_gradient = np.dot(grad, self.input.T)
        input_gradient = np.dot(self.weights.T, grad)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * grad
        return input_gradient
        
        
    def get_params(self, weights=True, bias=True):
        
        if weights and bias:
            return self.weights, self.bias
        elif weights:
            return self.weights
        elif bias:
            return self.bias
        else:
            raise TypeError("Pick a parameter to view")
      
        
    def __str__(self):
        return f"""Weights shape: {self.weights.shape}
Weights: {self.weights}
Bias shape: {self.bias.shape}
Bias: {self.bias}"""
        
        
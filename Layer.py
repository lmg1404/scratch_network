import numpy as np

np.random.seed(0)
class Layer:
    def __init__(self, output_shape: int, input_shape: int, initializiation="random"):
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
        
    def forward(self, input):
        # ensure the shapes are correct
        try:
            assert self.weights.shape[1] == input.shape[0]
        except:
            raise ValueError(f"Shapes do not match {self.weights.shape} != {input.shape}")
            
        return self.weights@input + self.bias
        
    
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
        
    def __str__(self):
        return f"""Weights shape: {self.weights.shape}
Weights: {self.weights}
Bias shape: {self.bias.shape}
Bias: {self.bias}"""
        
if __name__ == "__main__":
    layer = Layer(3, 4)
    print(layer)
    
    a = np.random.randn(4,3)
    print(a.shape)
    output = layer.forward(a)
    print(output)
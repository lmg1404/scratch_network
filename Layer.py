from abc import ABC, abstractmethod

class Layer(ABC):
    """
    A class to represent a layer (includes dense, activations) that has methods to go backwards and forwards

    Methods
    --------
    forward(inputs)
        forward propagation depending on the class and uses
    
    backward(inputs)
        backwards propagation also depending on the class using defined derivatives
    """
    
    @abstractmethod
    def forward(self):
        pass   
    
    @abstractmethod
    def backward(self):
        pass
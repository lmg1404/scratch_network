import numpy as np
from Dense import Dense

class Network:
    def __init__(self, layers=None, loss_func):
        self.layers = layers # list or nothing there
        self.loss_func = loss_func # class?
    
    def train(self, epochs : int, train_data, val_data, verbose=True):
        for i in range(epochs):
            for x, y in train_data:
                # TODO: train forward, maybe add minibatches to make faster?
                pass
            
            for x, y in val_data:
                # TODO: validation forward and loss
                pass
            
            if verbose:
                print(f"Epoch: {i+1} / Train loss: / Validation Loss: ")
    
    def predict(self):
        # TODO: just forward prop through the prediction examples
        pass
    
    def history(self):
        # TODO: get training history like in TF and PyTorch
        # think this is a good idea
        pass
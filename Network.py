import numpy as np
from Dense import Dense
from History import History

class Network:
    def __init__(self, loss_func, layers=None):
        self.layers = layers # list or nothing there
        self.loss_func = loss_func # class?
        self.history = History()
    
    def add(self, layers : list): # just adding a list makes everything much easier
        if self.layers:
            if type(self.layers) == list: # layers is a list no problem
                self.layers += layers # list plus list is still 1D list
            else: # layers is a dense object or some other object then we do something else
                self.layers = [self.layers] + layers # this should make it a list
        else:
            self.layers = layers
    
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
    
    def predict(self, test_data):
        # TODO: just forward prop through the prediction examples
        pass
    
    def history(self):
        # TODO: get training history like in TF and PyTorch
        return self.history
    
    
def to_list(arg):
    our_list = ['bug', 'wheat']
    print([arg] + our_list)
    
to_list(6)
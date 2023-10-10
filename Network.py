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
            # keep track of losses for overfitting if verbose/history
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            # train outr model
            for x, y in train_data:
                
                # compute forward
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                epoch_train_loss += self.loss_func.compute(output, y)
                
                # go backwards
                backwards = self.loss_func.compute_derivative(output, y)
                for layer in self.layer.reverse():
                    backwards = layer.backward(backwards)
            
            # average loss
            epoch_train_loss /= len(train_data)
                    
            # validation to make sure we're generalizing well
            for x, y in val_data:
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                epoch_val_loss += self.loss_func.compute(output, y)
                
            # average loss again
            epoch_val_loss /= len(val_data)
            
            # verbose so we get a terminal output
            if verbose:
                print(f"Epoch: {i+1} / Train loss: {epoch_train_loss} / Validation Loss: {epoch_val_loss}")
            
            # adding to history so we can plot if we want to by calling history
            self.history.add_epoch(i)
            self.history.add_train_metric(epoch_train_loss)
            self.history.add_val_metric(epoch_val_loss)
    
    def predict(self, test_data, verbose=True):
        
        # initialize loss to 0, go forward like before but not backwards
        test_loss = 0
        for x, y in test_data:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            test_loss += self.loss_func.compute(output, y)
                
        # average loss again
        test_loss /= len(test_data)
        
        self.history.add_test_metric(test_loss)
    
    def history(self):
        return self.history
    
    
    
import numpy as np
from Dense import Dense
from History import History

class Network:
    def __init__(self, loss_func, layers=None):
        self.layers = layers # list or nothing there
        self.loss_func = loss_func # class?
        self.history = History(str(loss_func))
    
    def add(self, layers : list): # just adding a list makes everything much easier
        if self.layers:
            if type(self.layers) == list: # layers is a list no problem
                self.layers += layers # list plus list is still 1D list
            else: # layers is a dense object or some other object then we do something else
                self.layers = [self.layers] + layers # this should make it a list
        else:
            self.layers = layers
    
    def train(self, epochs : int, x_train, y_train, learning_rate=0.01, verbose=True):
        for i in range(epochs):
            # keep track of losses for overfitting if verbose/history
            epoch_train_loss = 0
            
            # independent code's way of doing zip is way better than what i had before
            # train our model
            for x, y in zip(x_train, y_train):
                
                # compute forward
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                epoch_train_loss += self.loss_func.compute(output, y)
                
                # go backwards
                backwards = self.loss_func.compute_derivative(output, y)
                for layer in reversed(self.layers):
                    backwards = layer.backward(backwards, learning_rate)
            
            # average loss
            epoch_train_loss /= len(x_train)
            
            # verbose so we get a terminal output
            if verbose:
                print(f"Epoch: {i+1} / Train loss: {epoch_train_loss}")
            
            # adding to history so we can plot if we want to by calling history
            self.history.add_epoch(i)
            self.history.add_train_metric(epoch_train_loss)
    
    def predict(self, test_data):
        # I have no idea why I put that before
        # cleaner and the correct way for 1 training example
        output = test_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def plot(self):
        self.history.plot()
    
    
    
import numpy as np

class Binary_Cross_Entropy:
    def compute(self, y_hat, y):
        loss = 0
        epsilon = 1e-7 # here incase there are any 0s
        for i in range(len(y)):
            loss += - (y[i]*np.log10(y_hat[i] + epsilon) + (1-y[i])*np.log10(1-y_hat[i] + epsilon))
        return loss  
    
    def compute_derivative(self, y_hat, y):
        back = 0
        for i in range(len(y)):
            back += y[i]/y_hat[i] + (1-y[i])/(1-y_hat[i]) 
        return back
    
    def __str__(self):
        return "Binary Cross Entropy"
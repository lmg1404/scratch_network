import numpy as np

class Binary_Cross_Entropy:
    def compute(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-5, 1 - 1e-5)
        # had to fix, should be mean across given training example
        # log10 also didn't make any sense, I think the algo would prob freak out
        # again using IC since I couldn't come up with the correct answer the first time
        return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))
    
    def compute_derivative(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
        return (y/y_hat - (1-y)/(1-y_hat)) / np.size(y) # correct interpretation, it should NOT be a scalar!!!!
    
    def __str__(self):
        return "Binary Cross Entropy"
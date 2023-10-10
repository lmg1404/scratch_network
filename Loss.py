import numpy as np

class Binary_Cross_Entropy:
    def compute(self, inputs):
        pass
    
    def compute_derivative(self, inputs):
        pass
    
    def __str__(self):
        return "Binary Cross Entropy Loss"

def multi_class_cross_entropy(y, y_hat):
    # add the epsilon so there are no divide by 0 errors
    loss = 0
    epsilon = 1e-7
    for i in range(len(y)):
        dud = - (y[i]*np.log10(y_hat[i] + epsilon) + (1-y[i])*np.log10(1-y_hat[i] + epsilon))
        loss += dud
        print(dud)
    return loss

bce = Binary_Cross_Entropy()
print(str(bce))
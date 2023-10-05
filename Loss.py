import numpy as np

y = np.array([0, 0, 0, 1])
y_hat = np.array([0, 1, 0, 0])

def multi_class_cross_entropy(y, y_hat):
    # add the epsilon so there are no divide by 0 errors
    loss = 0
    epsilon = 1e-7
    for i in range(len(y)):
        dud = - (y[i]*np.log10(y_hat[i] + epsilon) + (1-y[i])*np.log10(1-y_hat[i] + epsilon))
        loss += dud
        print(dud)
    return loss

print(multi_class_cross_entropy(y,y_hat))
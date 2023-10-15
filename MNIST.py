from Dense import Dense
from Act_Func import Softmax, Sigmoid, ReLU, Tanh
from Loss import Binary_Cross_Entropy
from Network import Network
from keras.datasets import mnist
from keras.utils import to_categorical # different from tutorials online
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train[0])

def preprocess(x, y):
    x = x.reshape(x.shape[0], 28*28) # NN is flat, so i don't think a square will work
    y = to_categorical(y)
    return x, y

x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

layers = [Dense(100, 28*28, "he-et-al"), Sigmoid(), Dense(40, 100, "he-et-al"), Tanh(), Dense(10, 40, "he-et-al"), Softmax()]
loss = Binary_Cross_Entropy()

network = Network(loss, layers)
network.train(80, )
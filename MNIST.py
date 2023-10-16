from Dense import Dense
from Act_Func import Softmax, Sigmoid, ReLU, Tanh
from Loss import Binary_Cross_Entropy
from Network import Network
from keras.datasets import mnist
from keras.utils import to_categorical # different from tutorials online
import matplotlib.pyplot as plt
import numpy as np

print("Loading Dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(x, y):
    x = x.reshape(x.shape[0], 28*28, 1) # NN is flat, so i don't think a square will work, this gives a column (good)
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1) # gives a column, likely that the loss function won't work without it
    return x, y

print("Preprocessing data")
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

x_train = x_train/255
x_test = x_test/255

layers = [Dense(40, 28*28), Tanh(), Dense(10, 40), Softmax()]
loss = Binary_Cross_Entropy()

print("Randomly selecting training set")
train_size = 1000
np.random.seed(0)
random_selections = np.random.randint(0, x_train.shape[0], train_size)
x_train = x_train[random_selections]
y_train = y_train[random_selections]

network = Network(loss, layers)
network.train(20, x_train, y_train, 0.01, True)
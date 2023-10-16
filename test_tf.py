import tensorflow as tf
import keras
import numpy as np

model = keras.Sequential()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=28*28), 
    keras.layers.Dense(40, activation=tf.nn.tanh), 
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer="sgd",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_size = 1000
np.random.seed(0)
random_selections = np.random.randint(0, x_train.shape[0], train_size)
x_train = x_train[random_selections]
y_train = y_train[random_selections]
model.fit(x_train, y_train, epochs=10, verbose=2)
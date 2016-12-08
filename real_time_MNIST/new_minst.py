#%matplotlib inline

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train/255
X_test = X_test/255

y_dummy_train = to_categorical(y_train)
y_dummy_test = to_categorical(y_test)

#this is for the hinge loss:
#y_dummy_train = y_dummy_train*2 - 1


model = Sequential()
model.add(Convolution2D(24, 3, 3, input_shape = (1, 28, 28), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])




hist = model.fit(X_train, y_dummy_train,
                 validation_data = (X_test, y_dummy_test), nb_epoch=1, batch_size=128, verbose = 3)

model.save("/Users/alexpapiu/Documents/Data/CNN_weights/full_model.mnist")

#testing building a convnet on fingers:
import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam

from sklearn.cross_validation import train_test_split
%matplotlib inline
os.chdir("/Users/alexpapiu/Documents/Conv/OpenCV_CNN")
from webcam_cnn_pipeline import *

os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")
X = np.load("imgs.npy")

import keras



X.shape
model = return_compiled_model(input_shape = X.shape[1:])


"""a one-layer perceptron"""
input_shape = X.shape[1:]


model = Sequential()
model.add(MaxPooling2D((8,8), input_shape = input_shape))
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy",
              optimizer = adam(lr = 0.001), metrics = ["accuracy"])

model.summary()


X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify = y, random_state = 3)

model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch = 1, batch_size= 16)

model.predict(X_val)


def show_img(arr):
    plt.imshow(arr.transpose(1, 2, 0))

show_img(X[0])


n = int(X.shape[0]/2)
y = n*[0] + n*[1]



from sklearn import metrics
metrics.log_loss(y_tr, X_tr.shape[0]*[1])


model.predict(X_tr)


#dict for label:
labelz = {0:"alex", 1:"jon"}
real_time_pred(model, labelz, nframes = 500)

model.save("basic_model")
json_model = model.to_json()

json_model

model.save_weights("basic_model_weights")

test_var = 4123123

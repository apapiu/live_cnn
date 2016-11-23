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

X.shape
model = return_compiled_model(input_shape = X.shape[1:])


plt.imshow(X[190].transpose(1, 2, 0))
368/2
y = np.hstack((np.zeros(184), np.ones(184)))


X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify = y, random_state = 3)



model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch = 5, batch_size= 8)

model.predict(X_val)

model.summary()


#dict for label:
labelz = {0:"alex", 1:"jon"}
real_time_pred(model, labelz, nframes = 1000)

model.save("basic_model")
json_model = model.to_json()

json_model

model.save_weights("basic_model_weights")

test_var = 4123123

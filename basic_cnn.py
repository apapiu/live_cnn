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




os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")


X = np.load("images.npy")

X = X_1
X = X.transpose((0,3,1,2)) #for theano
X = X/255

X
X.shape

plt.imshow(X[10].transpose(1, 2, 0))

y = np.hstack((create_label(0, X_1.shape[0]),
               create_label(1, X_2.shape[0]),
               create_label(2, X_3.shape[0])))

y = create_label(0, X_1.shape[0])
pd.Series(y).value_counts()

y_ohe = to_categorical(y, nb_classes = 3)



X_tr, X_val, y_tr, y_val = train_test_split(X, y_ohe, stratify = y, random_state = 3)

def return_compiled_model():
    inp = X_tr.shape[1:]
    model = Sequential()
    model.add(MaxPooling2D((2,2), input_shape = inp))

    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), input_shape = inp))

    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2)))


    model.add(Flatten())

    model.add(Dense(128, activation = "relu"))

    #model.add(Dense(1, activation = "sigmoid"))
    model.add(Dense(3, activation = "softmax"))
    model.summary()

    model.compile(loss = "binary_crossentropy", optimizer = adam(lr = 0.0002), metrics = ["accuracy"])
    return(model)

model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch = 5, batch_size= 16)

model.fit(X, y_ohe, nb_epoch = 2, batch_size= 16)


model.predict_classes(X_val)

model.predict(X_val)

%time model.predict(X_val[:10])


model.save("basic_model")
json_model = model.to_json()

json_model

model.save_weights("basic_model_weights")

test_var = 4123123

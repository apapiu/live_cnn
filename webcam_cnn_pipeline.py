#functions:
import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam

from sklearn.cross_validation import train_test_split

font = cv2.FONT_HERSHEY_SIMPLEX

#captures video and saves an image to an array a few times/second:
def imgs_to_arr(cp, nr = 100, nframe = 10):
    imgs = []
    for i in range(nr*10):
        ret, frame = cp.read(0)
        if i < 75:
            cv2.putText(frame, "Prepare for Caption", (15,25),
                        font, 0.75, (200,255,155), 1, cv2.LINE_AA)
        if i % nframe == 0 and i > 75: #capture every n frames and leave a few frames to get in position
            imgs.append(frame)
            cv2.putText(frame, str(i/nframe), (15,25),
                        font, 0.75, (200,255,155), 1, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    imgs = np.array(imgs)
    return(imgs)


def create_label(number = 2, length = 100):
    twos = []
    for i in range(length):
        twos.append(number)
    twos = np.array(twos)
    return(twos)


def create_matrices(X_1, X_2):
    X = np.vstack((X_1, X_2))
    X = X.transpose(0, 3, 1, 2)
    X = X/255

    y = np.hstack((create_label(0, X_1.shape[0]),
                   create_label(1, X_2.shape[0])))
    return(X, y)


def return_compiled_model(input_shape):
    model = Sequential()
    model.add(MaxPooling2D((3,3), input_shape = (3, 144, 256)))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy",
                  optimizer = adam(lr = 0.001), metrics = ["accuracy"])
    return(model)

def return_compiled_model_2(input_shape):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation = "relu", input_shape = (3, 144, 256)))
    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2)))

    #model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(Dropout(0.5))
    #model.add(GlobalAveragePooling2D())

    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy", optimizer = adam(lr = 0.001), metrics = ["accuracy"])
    return(model)


def predict_from_frame(model, frame, labelz):
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, 0)
    frame = frame/255
    preds  = model.predict_classes(frame, verbose = False)[0][0]
    label = labelz[preds]
    return(label)


def real_time_pred(model, labelz, nframes = 1000):
    cp = cv2.VideoCapture(0)
    cp.set(3, 256)
    cp.set(4, 144)
    for i in range(nframes):
        ret, frame = cp.read(0)
        if i % 10 == 0:
            label = predict_from_frame(model, frame, labelz)

        cv2.putText(frame, label, (15,25), font, 0.75, (200,255,155), 1, cv2.LINE_AA)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

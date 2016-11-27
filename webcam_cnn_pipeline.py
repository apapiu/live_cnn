#functions:
import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D
from keras.layers import Flatten, GlobalAveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam

from sklearn.cross_validation import train_test_split

font = cv2.FONT_HERSHEY_SIMPLEX

def imgs_to_arr(cp, nr = 100, nframe = 10):
    """captures video and saves an image to an array a few times/second"""

    imgs = []
    for i in range(nr*10):
        ret, frame = cp.read(0)
        if i < 75:
            cv2.putText(frame, "Prepare for Caption", (15,25),
                        font, 0.75, (200,255,155), 1, cv2.LINE_AA)
        #capture every n frames and leave a few frames to get in position
        if i % nframe == 0 and i > 75:
            imgs.append(frame)
            cv2.putText(frame, str(i/nframe), (15,25),
                        font, 0.75, (200,255,155), 1, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    imgs = np.array(imgs)
    return(imgs)

def create_labels(sizes):
    """create labels starting at 0 with specified sizes

    sizes must be a tuple
    """

    labels = []
    for i, size in enumerate(sizes):
        labels += size*[i]
    return(labels)


def create_matricez(*args):
    """ create X and label matrices from a bunch of X matrices"""

    X = np.vstack(args)
    X = X.transpose(0, 3, 1, 2)
    X = X/255

    sizes = (X.shape[0] for X in args)
    y = create_labels(sizes)
    return(X, y)

#to be deleted since it only takes two matrices
def create_matrices(X_1, X_2):
    X = np.vstack((X_1, X_2))
    X = X.transpose(0, 3, 1, 2)
    X = X/255

    sizes = (X_1.shape[0], X_2.shape[0])
    y = create_labels(sizes)

    return(X, y)


def return_compiled_model(input_shape):
    """a one-layer perceptron"""

    model = Sequential()
    model.add(MaxPooling2D((6,6), input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy",
                  optimizer = adam(lr = 0.001), metrics = ["accuracy"])
    return(model)

def return_compiled_model_2(input_shape):
    """a 2-layer convnet"""

    model = Sequential()
    model.add(MaxPooling2D((2,2), input_shape = input_shape))
    model.add(Convolution2D(32, 3, 3, activation = "relu", ))
    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3,3)))

    #model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(Convolution2D(64, 3, 3, activation = "relu"))
    #model.add(Dropout(0.5))
    #model.add(GlobalAveragePooling2D())

    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy",
                  optimizer = adam(lr = 0.001), metrics = ["accuracy"])
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
    cp.set(3, 2*256)
    cp.set(4, 2*144)
    for i in range(nframes):
        ret, frame = cp.read(0)
        if i % 10 == 0:
            label = predict_from_frame(model, frame, labelz)

        cv2.putText(frame, label, (15,25), font, 0.75, (200,255,155), 1, cv2.LINE_AA)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

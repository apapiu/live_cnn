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



def annotate(frame, label):
    """writes label on image"""

    cv2.putText(frame, label, (20,30), font,
                fontScale = 1,
                color = (100,255,155),
                thickness =  1,
                lineType =  cv2.LINE_AA)

def imgs_to_arr(cp, nr = 100, nframe = 5):
    """captures video and saves an image to an array a few times/second"""

    imgs = []
    for i in range(nr*10):
        ret, frame = cp.read(0)
        if i < 75:
            annotate(frame, "Prepare for Caption")
        #capture every n frames and leave a few frames to get in position
        if i % nframe == 0 and i > 75:
            imgs.append(frame)
            annotate(frame, str(i/nframe))

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


def create_matrices(array_list):
    """ create X and label matrices from a bunch of numpy arrays"""

    X = np.vstack(array_list)
    X = X.transpose(0, 3, 1, 2)
    X = X/255

    sizes = (X.shape[0] for X in array_list)
    y = create_labels(sizes)
    return(X, y)



def return_compiled_model(input_shape, num_class = 2):
    """a one-layer perceptron"""

    model = Sequential()
    model.add(MaxPooling2D((8,8), input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_class, activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = adam(lr = 0.001), metrics = ["accuracy"])
    return(model)



def return_compiled_model_2(input_shape):
    """a 2-layer convnet"""

    model = Sequential()
    model.add(MaxPooling2D((3,3), input_shape = input_shape))

    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy",
                  optimizer = adam(lr = 0.001), metrics = ["accuracy"])
    return(model)





def predict_from_frame(model, frame, labelz, resize = False, input_shape = (128, 128)):
    """takes a frame and outputs a class prediction

    Parameters:
    -----------
    model: a keras or sklearn model
    frame: a frame given as an numpy array
    labelz: dictionary of classes for example {0:"cat", 1:"dog"}
    resize: Boolean indicating if the image should be reshaped before
        being fed into the model
    input shape - 2-dimensional tuple giving the height and width
        the image should be reshaped to - this should be the shape
        accepted by the model

    """

    if resize:
        frame = cv2.resize(frame, input_shape)

    frame = frame.transpose((2, 0, 1)) #this because we use theano shape defaults
    frame = np.expand_dims(frame, 0)
    frame = frame/255.0

    preds  = model.predict_classes(frame, verbose = False)[0]
    label = labelz[preds]
    return(label)


def real_time_pred(model, labelz, cp, nframes = 1000, resize = False,
                   input_shape = (128, 128)):

    for i in range(nframes):
        ret, frame = cp.read(0)

        #predict every 10 frames:
        if i % 10 == 0:
            label = predict_from_frame(model, frame, labelz,
                                       resize = resize,
                                       input_shape = input_shape)

        annotate(frame, label)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

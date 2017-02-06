#functions:
#to get the dependencies create a conda environment:
#conda create --name live_cnn keras ipython opencv scikit-learn
#note that the dim ordering here is the theano one - it's faster on CPU.

#TODO: do some detection/ bounding box CNN's?

import time
import os
import cv2
import numpy as np

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam

from sklearn.cross_validation import train_test_split

font = cv2.FONT_HERSHEY_SIMPLEX

font = cv2.FONT_ITALIC

def annotate(frame, label, size = 1):
    """writes label on image"""
    cv2.putText(frame, label, (20,30), font,

                fontScale = size,
                color = (255, 255, 0),
                thickness =  1,
                lineType =  cv2.LINE_AA)

def imgs_to_arr(cp, label, nr = 100, nframe = 20):
    """captures video and saves an image to an array a few times/second"""

    imgs = []
    #range is made so that nr is actually the number of photos taken:
    for i in range(int(nr)*nframe + 80):
        ret, frame = cp.read(0)
        if i < 75:
            annotate(frame, "Prepare: {0}".format(label), size = 0.7)
        #capture every n frames and leave a few frames to get in position:
        if i % nframe == 0 and i > 70:
            imgs.append(frame)
            print((i - 70)/nframe)
            #annotate(frame, str((i - 70)/nframe))

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

    #TODO: create a time based validation - take the last 15% percent of data and make it val
    return(X, y)

def return_compiled_model(input_shape, num_class = 2):
    """a one-layer perceptron"""

    model = Sequential()
    model.add(MaxPooling2D((8,8), input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_class, activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = adam(lr = 0.001),
                  metrics = ["accuracy"])
    return(model)


def return_compiled_model_2(input_shape, num_class = 2):
    """a 3-layer convnet"""

    model = Sequential()
    #max pooling at first since images are big and
    #also for small translation invariance:
    model.add(MaxPooling2D((3,3), input_shape = input_shape))

    model.add(Convolution2D(32, 3, 3, activation = "relu", border_mode = "same"))
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, 3, 3, activation = "relu", border_mode = "same"))
    model.add(MaxPooling2D((2,2)))

    model.add(Convolution2D(64, 3, 3, activation = "relu", border_mode = "same"))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_class, activation = "softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return(model)

#model = return_compiled_model_2(input_shape = (3, w, h), num_class = 5)
#model.summary()


os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")

model = load_model("basic_model")

feat_extr = Model(input = model.input, output = model.get_layer("dropout_1").output)


#TODO: make this accept arbitrarily sized input:
def pre_trained_model(num_class, lr = 0.0005):
    """
    returns a pretrained model adding a fully connected
    layer on top for prediction with as many final units
    as number of classes
    """

    os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")
    model = load_model("basic_model")

    feat_extr = Model(input = model.input, output = model.get_layer("dropout_1").output)


    output = Dense(num_class, activation = "softmax", name = "bleh")(feat_extr.output)
    new_model = Model(input = feat_extr.input, output = output)

    #make all layers except last not trainable:
    for layer in new_model.layers[:-1]:
       layer.trainable = False
       new_model.layers[-1].trainable = True

    new_model.compile(loss = "categorical_crossentropy",
                      optimizer = adam(lr = lr), metrics = ["accuracy"])

    return new_model


def predict_from_frame(model, frame, labelz, resize = False, input_shape = (128, 128)):
    """takes a frame and outputs a class prediction

    Parameters:
    -----------
    model: a keras or sklearn model
    frame: a frame given as an numpy array
    labelz: dictionary of classes for example {0:"cat", 1:"dog"}
    resize: Boolean indicating if the image should be reshaped before
        being fed into the model
    input shape: 2-dimensional tuple giving the height and width
        the image should be reshaped to - this should be the shape
        accepted by the model

    """

    if resize:
        frame = cv2.resize(frame, input_shape)

    frame = frame.transpose((2, 0, 1)) #this because we use theano shape defaults
    frame = np.expand_dims(frame, 0)
    frame = frame/255.0

    #preds  = model.predict_classes(frame, verbose = False)[0]
    preds = np.argmax(model.predict(frame, verbose = False)[0])

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

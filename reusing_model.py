import os
from keras.models import load_model
import cv2
import sys

sys.path.append("/Users/alexpapiu/Documents/Conv/OpenCV_CNN")
from webcam_cnn_pipeline import *

os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")

model = load_model("basic_model")

h= 1.5*144
w = 2*144


labelz = dict(enumerate(["happy", "sad", "normal"]))

#open a new video:
cp = cv2.VideoCapture(0)
cp.set(3, h)
cp.set(4, w)

real_time_pred(model, labelz, cp = cp, nframes = 10000)

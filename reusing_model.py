import sys
import os
import cv2

from keras.models import load_model
sys.path.append("/Users/alexpapiu/Documents/Conv/OpenCV_CNN")
from webcam_cnn_pipeline import return_compiled_model_2, real_time_pred


model_name = sys.argv[1]
w = 1.5*144
h = 2*144

#keep track of all labels:
all_labels = {"model_hand":["A", "B", "C", "D", "No Hand"],
              "basic_model":["happy", "sad", "normal", "incredulous"],
              "model_face":["happy", "sad", "normal"]}

labelz = dict(enumerate(all_labels[model_name]))


os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")
model = return_compiled_model_2(input_shape = (3,int(h),int(w)),
                                        num_class = len(labelz))
model.load_weights(model_name)
#open a new video:
cp = cv2.VideoCapture(0)
cp.set(3, w)
cp.set(4, h)

real_time_pred(model, labelz, cp = cp, nframes = 10000)

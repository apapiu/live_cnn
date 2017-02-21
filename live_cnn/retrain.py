#continue training on mew images:
import time
import os
from keras.models import load_model
import cv2
import sys

from keras.optimizers import adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

module_folder = "/Users/alexpapiu/Documents/Conv/OpenCV_CNN"

sys.path.append(module_folder)
from webcam_cnn_pipeline import *

N = int(sys.argv[1])
model_name = sys.argv[2]

w = 1.5*144
h = 2*144

aug_lvl = 0.15

lr = input("What should be the learning rate - 0.001 to 0.0001? ")

#keep track of all labels:
all_labels = {"model_hand":["A", "B", "C", "D", "No Hand"],
              "basic_model":["happy", "sad", "normal", "incredulous"],
              "model_face":["happy", "sad", "normal"]}

labelz = dict(enumerate(all_labels[model_name]))
num_classes = len(labelz)

data_folder = "/Users/alexpapiu/Documents/Data/OpenCV_CNN"
os.chdir(data_folder)
model = load_model(model_name)

model.compile(loss = "categorical_crossentropy",
              optimizer = adam(lr = lr), #smaller learning rate.
              metrics = ["accuracy"])


#open a new video:
cp = cv2.VideoCapture(0)
cp.set(3, w)
cp.set(4, h)

#getting new data:
data = []
for i in range(int(num_classes)):
    current_label = labelz[i]
    print("Label {0}".format(current_label))
    X = imgs_to_arr(cp = cp, nr = N, nframe = 20, label = current_label)
    data.append(X)
    time.sleep(3)


cp.release()
cv2.destroyAllWindows()

X, y = create_matrices(data)
y_ohe = to_categorical(y)


#a terrible way to save the images:
#use clock to differentiate it from the other saved imgs:
#will need to concat the arrays to retrain:
clock = time.clock()
location = "{0}/{1}_/".format(data_folder, model_name)
np.save(location + "X" + str(clock), X)
np.save(location + "y" + str(clock), y)


#do image augmentation:
datagen = ImageDataGenerator(
        shear_range=aug_lvl,
        zoom_range=aug_lvl,
        rotation_range=aug_lvl*10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=aug_lvl,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=aug_lvl,
        fill_mode='nearest')

datagen.fit(X)

print("training model")
model.fit_generator(datagen.flow(X, y_ohe, batch_size=32),
                        samples_per_epoch=len(X)*5,
                        nb_epoch=10)


more_training = input("Should I train the model more? Answer n or a number indicating \
                       the numbers of additional epochs: ")

if more_training != "n":
    model.fit_generator(datagen.flow(X, y_ohe, batch_size=32),
                            samples_per_epoch=3*len(X),
                            nb_epoch = int(more_training))

print("saving model")
model.save(model_name)

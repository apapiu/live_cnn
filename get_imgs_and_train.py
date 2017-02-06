#code to get images from open cv into python
#webcam_cnn_pipeline is a local module so export path first
import sys
from keras.preprocessing.image import ImageDataGenerator

sys.path.append("/Users/alexpapiu/Documents/Conv/OpenCV_CNN")
from webcam_cnn_pipeline import *

w = 1.5*144
h = 2*144

#setting up camera:
cp = cv2.VideoCapture(0)
cp.set(3, w)
cp.set(4, h)

os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")

#N - number of images
#aug_ct how many augmented training imgs per real image
#10 10 is the least for basic tasks, at least 20 10 for tricky stuff
N = int(sys.argv[1])
aug_ct = int(sys.argv[2])

num_classes = int(input("How many classes do you want? "))
labels = []
for i in range(1, num_classes + 1):
    newlabel = input("What is label {0}: ".format(i))
    labels.append(newlabel)

labelz = dict(enumerate(labels))

save_imgs = "n" #input("Do you want to save the images?[y/n]")
model_type = "conv" #input("What type of model do you want? Choose conv or mlp ")

pretrained = input("Shoud I load a pretrained model? [y/n]")
model_name = input("What should I save the new model as? Press n for no save: ")

#gathering the data:
data = []
for i in range(int(num_classes)):
    current_label = labelz[i]
    print("Get ready for label {0}".format(current_label))
    X = imgs_to_arr(cp = cp, nr = N, nframe = 50, label = current_label)
    data.append(X)
    time.sleep(3)

cp.release()
cv2.destroyAllWindows()

X, y = create_matrices(data)

#one hot encoded labels for keras:
y_ohe = to_categorical(y)

if save_imgs == "y":
    np.save("imgs", X)
    np.save("labels", y)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#predicting real time using a keras model:

inp = X.shape[1:]

if model_type == "mlp":
    model = return_compiled_model(input_shape = inp, num_class = num_classes)
elif model_type == "conv" and pretrained == "n":
    model = return_compiled_model_2(input_shape = inp, num_class = num_classes)
elif model_type == "conv" and pretrained == "y":
    model = pre_trained_model(num_class = num_classes)

#X_tr, X_val, y_tr, y_val = train_test_split(X, y_ohe, stratify = y,
#                                            random_state = 3, test_size = 0.1) #0.15)
X_tr, y_tr = X, y_ohe


print(X_tr.shape)

print("training model")

datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.15,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest')

#if using pretrained model, train last FC layer only for a bit:
if pretrained == "y":
    print("train only with last FC layer unfrozen")
    model.fit(X_tr, y_tr, nb_epoch=10, batch_size=32, verbose = 0)

    #unfreeze more top layers and train more:
    print("Unfreezing More Layers:")
    model.get_layer("dense_1").trainable = True
    model.get_layer("convolution2d_2").trainable = True
    model.get_layer("convolution2d_3").trainable = True

    model.compile(loss = "categorical_crossentropy",
                      optimizer = adam(lr = 0.0006), metrics = ["accuracy"])

    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=32),
        #validation_data = (X_val, y_val),
        samples_per_epoch=aug_ct*len(X_tr), nb_epoch=5)

else:
    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=32),
        #validation_data = (X_val, y_val),
        samples_per_epoch=aug_ct*len(X_tr), nb_epoch=5)


more_training = input("Should I train the model more? Answer n or a number indicating \
                       the numbers of additional epochs: ")

if more_training != "n":
    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=24),
    #validation_data = (X_val, y_val),
    samples_per_epoch=2*len(X_tr), nb_epoch = int(more_training))
    # model.fit(X_tr, y_tr, validation_data = (X_val, y_val),
    #           nb_epoch=int(more_training), batch_size=24)


if model_name != "n":
    model.save(model_name)

#open a new video
cp = cv2.VideoCapture(0)
cp.set(3, w)
cp.set(4, h)

real_time_pred(model, labelz, cp = cp, nframes = 10000)

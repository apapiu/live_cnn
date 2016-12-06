#code to get images from open cv into python
#webcam_cnn_pipeline is a local module so export path first
#!export PYTHONPATH="$PYTHONPATH:/Users/alexpapiu/Documents/Conv/OpenCV_CNN"
from webcam_cnn_pipeline import *
import sys
#setting up camera:
cp = cv2.VideoCapture(0)
cp.set(3, 2*256)
cp.set(4, 2*144)

os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")

N = int(sys.argv[1])

num_classes = int(input("How many classes do you want?"))

labels = []
for i in range(1, num_classes + 1):
    newlabel = input("What is label {0}".format(i))
    labels.append(newlabel)

labelz = dict(enumerate(labels))

save_imgs = input("Do you want to save the images?[y/n]")
model_type = input("What type of model do you want? Choose conv or mlp ")

if model_type not in ["conv", "mlp"]:
    raise ValueError("model_type must be conv for convolutional nets or mlp \
                      for feed-forward nets")

#gathering the data:
data = []
for i in range(int(num_classes)):
    current_label = labelz[i]
    print("Get ready for label {0}".format(current_label))
    X = imgs_to_arr(cp = cp, nr = N, nframe = 5)
    data.append(X)
    time.sleep(3)

cp.release()
cv2.destroyAllWindows()

X, y = create_matrices(data)

#one hot encoded labels for keras:
y_ohe = to_categorical(y)

if save_imgs == "y":
    np.save("imgs", X)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#predicting real time using a keras model:

inp = X.shape[1:]

if model_type == "mlp":
    model = return_compiled_model(input_shape = inp, num_class = num_classes)
elif model_type == "conv":
    model = return_compiled_model_2(input_shape = inp)

X_tr, X_val, y_tr, y_val = train_test_split(X, y_ohe, stratify = y,
                                            random_state = 3, test_size = 0.15)

print("training model")
model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch=3, batch_size=16)

model.save("basic_model")

more_training = input("Should I train the model more? Answer N or a number indicating \
                       the numbers of additional epochs: ")

if more_training != "N":
    model.fit(X_tr, y_tr, validation_data = (X_val, y_val),
              nb_epoch=int(more_training), batch_size=8)

#open a new video:
cp = cv2.VideoCapture(0)
cp.set(3, 2*256)
cp.set(4, 2*144)

real_time_pred(model, labelz, cp = cp, nframes = 10000)

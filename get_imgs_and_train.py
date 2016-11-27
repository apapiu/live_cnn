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

label1 = input("What is the first label? ")
label2 = input("What is the second label? ")
save_imgs = input("Do you want to save the images?[y/n]")
model_type = input("What type of model do you want? Choose conv or mlp ") #conv or mlp

if model_type not in ["conv", "mlp"]:
    raise ValueError("model_type must be conv for convolutional nets or mlp \
                      for feed-forward nets")

#creating datasets:
print("First Label")
X_1 = imgs_to_arr(cp = cp, nr = N, nframe = 5)
X_1.shape
time.sleep(5)

print("Second Label")
X_2 = imgs_to_arr(cp = cp, nr = N, nframe = 5)
#X_3 = imgs_to_arr(cp = cp, nr = 200, nframe = 10)

#X, y = create_matrices(X_1, X_2)
X, y = create_matricez(X_1, X_2)


if save_imgs == "y":
    np.save("imgs", X)
#X = np.load("imgs.npy")

#plt.imshow(X[70].transpose(1,2, 0))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#predicting real time using a keras model:

inp = X.shape[1:]

if model_type == "mlp":
    model = return_compiled_model(input_shape = inp)
elif model_type == "conv":
    model = return_compiled_model_2(input_shape = inp)

#model.summary()
#try:
#    model.load_weights("basic_model_weights")
#except:
#    pass

X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify = y,
                                            random_state = 3, test_size = 0.15)

print("training model")
model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch=5, batch_size=8)
model.save("basic_model")


#dict for label:
labelz = {0:label1, 1:label2}
real_time_pred(model, labelz, nframes = 10000)

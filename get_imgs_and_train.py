#code to get images from open cv into python
#webcam_cnn_pipeline is a local module so export path first
#!export PYTHONPATH="$PYTHONPATH:/Users/alexpapiu/Documents/Conv/OpenCV_CNN"
from webcam_cnn_pipeline import *

#setting up camera:
cp = cv2.VideoCapture(0)
cp.set(3, 256)
cp.set(4, 144)

os.chdir("/Users/alexpapiu/Documents/Data/OpenCV_CNN")

N = 100

#creating datasets:
print("First Label")
X_1 = imgs_to_arr(cp = cp, nr = N, nframe = 5)

time.sleep(5)

print("Second Label")
X_2 = imgs_to_arr(cp = cp, nr = N, nframe = 5)
#X_3 = imgs_to_arr(cp = cp, nr = 200, nframe = 10)

X, y = create_matrices(X_1, X_2)

#np.save("imgs", X)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#predicting real time using a keras model:

inp = X.shape[1:]
model = return_compiled_model(input_shape = inp)

#try:
#    model.load_weights("basic_model_weights")
#except:
#    pass

X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify = y, random_state = 3, test_size = 0.15)

print("training model")
model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch=5, batch_size=32)
model.save("basic_model")

model
#dict for label:
labelz = {0:"Open", 1:"Closed"}
real_time_pred(model, labelz, nframes = 1000)

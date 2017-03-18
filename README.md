###  How to train your cnn - Training Conv Nets live using keras and Open CV

Train Convnets using your webcam.

Train a CNN from scratch with 10 examples per class, every image gets augmented 5 times:

    python get_imgs_and_train.py 10 5


Reuse previous model:
  
    python reusing_model.py model_face
    
Retrain previous model with 5 new images per class:

    python retrain.py 5 model_face

model_face - a basic facial expression recognizer: happy, sad, normal

model_hand - classes A,B,C,D,No Hand



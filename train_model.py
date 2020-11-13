# Author: Karshiev Sanjar
# MSP Lab, KNU, 2020

#--------------# Training Mask detection model#---------------#


'''Recommendation: Read and follow the comments for each part of the code'''

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow.keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.interpolate
import os
from tensorflow.keras.layers import Conv2D

from keras.backend import sigmoid

#Deifining SWISH activation function for later usage
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish)})

#exclude warnings
import warnings
warnings.filterwarnings("ignore")

# Argument parser, the main arguments used in the code
ap = argparse.ArgumentParser()
# dataset path
ap.add_argument("-d", "--dataset", type = str, default = "dataset")

# path to output loss/accuracy plot
ap.add_argument("-p", "--plot", type=str, default="plot.png")

# path to output face mask detection model
ap.add_argument("-m", "--model", type=str,
	default="mask_detection.model")

args = vars(ap.parse_args())

'''1. Initialize learning rate. This is important factor that effects 
    the speed of learning
   2. Number of epochs
   3. Determining Batch-size. 
    In each training step "trainingX/Batch-size" number of images
    are trained. It effects also the training performance'''
    
INIT_LR = 1e-4
EPOCHS = 40
BS = 128

# Defining the list of images in the given dataset
print("->->-> ... loading images from the dataset ...")
imagePaths = list(paths.list_images(args["dataset"]))

# Creating empty list for data and labels
data = []
labels = []

# loop over the images
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# setting image size into (224,224), convert the image into array, and preprocessing the images
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding to categorize the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# split the dataset to the training and validation sets. 
# Here we are using 80 % of dataset for training and 
# and other 20 % is for validation set

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Data augmentation for training step
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# applying transfer learning from MobileNet V2 network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Model constructing
''' Here MobileNetV2 NN model is used as a base model for our network 
    We can change the architecture of headModel to increase the accuracy
    Refer to the Keras documenation to apply different convolution layers, 
    Pooling  and activation functions'''

headModel = baseModel.output
#Add Conv2D layer
#input_shape=(4,224,224,3)
#headModel= Conv2D(filters=5,kernel_size=(5,5),padding='same',activation='relu',input_shape=input_shape[1:])(headModel)
input_shape=(BS,224,224,3)
headModel= Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=input_shape[1:])(headModel)
headModel = swish(headModel)
headModel = MaxPooling2D(pool_size=(5, 5),padding='same')(headModel)
headModel= Conv2D(filters=32,kernel_size=(5,5),padding='same',input_shape=input_shape[1:])(headModel)
headModel = swish(headModel)
headModel = MaxPooling2D(pool_size=(5, 5),padding='same')(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128)(headModel)
headModel = swish(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Determining callback parameters for training
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.05, patience=10, min_lr=1e-7, verbose=1)
]

# compile our model
print("->->-> ... Compiling model ...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network for mask detection
print("->->-> ... Training model ...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
    callbacks = callbacks,
	epochs=EPOCHS)

# make predictions on the testing set
print("->->-> ... Evaluating network ...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Show Precision, Recall, f1-score and the number of images used in validation test
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("->->-> ... Saving mask detector model...")
model.save(args["model"], save_format="h5")

#save the model for testing on single image
model.save('face_mask.h5')

# load the trained model 
new_model = tensorflow.keras.models.load_model('face_mask.h5')

# test the trained & loaded model
from random import randint
i = randint(0, 552)
m = new_model.predict(testX[i].reshape(-1,224,224,3)) == new_model.predict(testX[i].reshape(-1,224,224,3)).max()
testX[i].shape

# showing the tested image
plt.imshow(testX[i][:,:,::-1])

# printing the result of prediction
print(np.array(['With Mask','Without Mask'])[m[0]])

# plot the training loss and accuracy
'''If you are using spider, the plots appear in 'Plots' part'''

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

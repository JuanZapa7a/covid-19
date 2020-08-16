# USAGE
# python train.py

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='../datasets/dataset',
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="output/covid.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-o", "--output", default="output",
                help="path to output .png loss/acc plot")
ap.add_argument("-m", "--model", type=str, default="vgg16",
                help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
  "vgg16": VGG16,
  "vgg19": VGG19,
  "inception": InceptionV3,
  "xception": Xception,  # TensorFlow ONLY
  "resnet": ResNet50
}

# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
  raise AssertionError("The --model command line argument should "
                       "be a key in the `MODELS` dictionary(vgg16,vgg19,"
                       "inception,xception,resnet)")

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
data = []
labels = []



# Creating Data (X) y labels (y) from imagepaths
# loop over the image paths
for imagePath in imagePaths:
  # extract the class label from the filename
  label = imagePath.split(os.path.sep)[-2]

  # load the image, swap color channels, and resize it to be a fixed
  # 224x224 pixels while ignoring aspect ratio
  image = cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (224, 224))
  # if we are using the InceptionV3 or Xception networks, then we
  # need to set the input shape to (299x299) [rather than (224x224)]
  # and use a different image processing function
  if args["model"] in ("inception", "xception"):
    image = cv2.resize(image, (299, 299))

  # update the data and labels lists, respectively
  data.append(image)
  labels.append(label)

# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 255]
data = np.array(data) / 255.0
labels = np.array(labels)

# perform one-hot encoding on the labels [0 1] normal [1 0] covid
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
aug = ImageDataGenerator(
  rotation_range=15, width_shift_range=0.1,
  height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
  horizontal_flip=False, fill_mode="nearest")

# load the MODELS[args["model"]] network, ensuring the head FC layer sets are left
# off
baseModel = MODELS[args["model"]](weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))
if args["model"] in ("inception", "xception"):
  baseModel = MODELS[args["model"]](weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(299, 299, 3)))

## TRANSFER LEARNING: feature extraction and fine-tuning

# When performing transfer learning feature extraction, we treat the
# pre-trained network as an arbitrary feature extractor, allowing the input
# image to propagate forward, stopping at pre-specified layer, and taking the
# outputs of that layer as our features.
# A standard machine learning classifier (for example, Logistic Regression),
# was trained on top of the CNN features, exactly as we would do with
# hand-engineered features such as SIFT, HOG, LBPs, etc.

# But there is another type of transfer learning, one that can actually
# outperform the feature extraction method. This method is called fine-tuning and requires us to perform “network surgery”.

# First, we take a scalpel and cut off the final set of fully connected
# layers (i.e., the “head” of the network where the class label predictions
# are returned) from a pre-trained CNN (typically VGG, ResNet, or Inception).
# We then replace the head with a new set of fully connected layers with random initializations.
# From there, all layers below the head are frozen so their weights cannot be
# updated (i.e., the backward pass in back propagation does not reach them).
# We then train the network using a very small learning rate so the new set
# of fully connected layers can learn patterns from the previously learned
# CONV layers earlier in the network — this process is called allowing the FC
# layers to “warm up”.
# Optionally, we may unfreeze the rest of the network and continue training.
# Applying fine-tuning allows us to utilize pre-trained networks to recognize
# classes they were not originally trained on.
# And furthermore, this method can lead to higher accuracy than transfer
# learning via feature extraction.

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
  layer.trainable = False
p = [args["output"], "{}_arch_{}.png".format(args["model"], os.getpid())]
print("[INFO] writting architecture model...")
from tensorflow.keras.utils import plot_model
plot_model(model, to_file=os.path.sep.join(p), show_shapes=True, dpi = 600)

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 25
BS = 8

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
  aug.flow(trainX, trainY, batch_size=BS),
  steps_per_epoch=len(trainX) // BS,
  validation_data=(testX, testY),
  validation_steps=len(testX) // BS,
  epochs=EPOCHS)


# make predictions on the testing set
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
# Classification report
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
report = classification_report(testY.argmax(axis=1), predictions.argmax(
  axis=1), target_names=lb.classes_)
print(report)
# Confusion Matrix
# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
#confmatrix = confusion_matrix(testY.argmax(axis=1), predictions)
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# save the classification report and confusion matrix to file
print("[INFO] writting classification report and confussion matrix text "
      "file...")
p = [args["output"], "{}_report_confmatrix{}.txt".format(args["model"], os.getpid())]
f = open(os.path.sep.join(p), "w")
f.write(report)
f.write(str(cm))
f.write(str(total))
f.write(str(acc))
f.write(str(sensitivity))
f.write(str(specificity))
f.close()

# save the history of training
print("[INFO] writting the history of training")
p = [args["output"], "{}_H_{}.cpickle".format(args["model"], os.getpid())]
with open(os.path.sep.join(p), 'wb') as f_pickle:
  pickle.dump(H.history, f_pickle)

# load the history of training
history = pickle.load(open(os.path.sep.join(p), 'rb'))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"][0:N], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"][0:N], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"][0:N], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"][0:N], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset with {}_{}".format(
  args["model"], os.getpid()))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
p = [args["output"], "{}_{}.png".format(args["model"],os.getpid())]
plt.savefig(os.path.sep.join(p))
plt.show()

# # serialize the model to disk
# print("[INFO] saving COVID-19 detector model...")
# model.save(args["model"], save_format="h5")
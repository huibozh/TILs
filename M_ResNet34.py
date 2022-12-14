# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:58:42 2022

@author: Huibo Zhang
"""

"""
#####ResNet-34
"""
import sys
assert sys.version_info >= (3, 5)

IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import matplotlib.pyplot as plt
"""
if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
    if IS_KAGGLE:
        print("Go to Settings > Accelerator and select GPU.")
"""
import numpy as np
import os
import glob
import cv2
import tensorflow as tf

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

from functools import partial

"""
###########ResNet-34 model
"""

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]
    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
resnet34_model = keras.models.Sequential()
resnet34_model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                        input_shape=[224, 224, 3]))
resnet34_model.add(keras.layers.BatchNormalization())
resnet34_model.add(keras.layers.Activation("relu"))
resnet34_model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    resnet34_model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
resnet34_model.add(keras.layers.GlobalAvgPool2D())
resnet34_model.add(keras.layers.Flatten())
resnet34_model.add(keras.layers.Dense(3, activation='softmax'))
resnet34_model.summary()


"""
####data imput
"""
##### Training set:
#Read input images and assign labels based on folder names
print(os.listdir("training/train/"))

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
SIZE = 224  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("training/train/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
X_train = np.array(train_images)
X_train = X_train / 255.0
y_train = np.array(train_labels)
len(train_images)
#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(train_labels) 
Y_train = le.transform(train_labels)

#####validation set
print(os.listdir("training/validation/"))
SIZE = 224  #Resize images
#Capture training data and labels into respective lists
val_images = []
val_labels = [] 

for directory_path in glob.glob("training/validation/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_images.append(img)
        val_labels.append(label)

#Convert lists to arrays        
X_val = np.array(val_images)
X_val = X_val / 255.0
y_val = np.array(val_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(y_val) 
Y_val = le.transform(y_val)


##
resnet34_model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy",metrics = ['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint('resnet34_model_for_classification.h5', verbose=1, save_best_only=True)
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer]

history = resnet34_model.fit(X_train,Y_train,
                    epochs=50, 
                    validation_data=(X_val,Y_val),
                    #validation_split=0.2,
                    verbose = 1,
                    callbacks=callbacks)


"""
### accuracy and loss plot
"""
accu = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accu))

plt.plot(epochs,accu, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='validation accuracy')
plt.title('Training and validation set accuracy')
plt.legend(loc='lower right')
plt.savefig('./plot/resnet34_accuracy.pdf', dpi = 1000)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./plot/resnet34_loss.pdf', dpi = 1000)
plt.show()


"""
#######  confusion matirx   ############
"""
#####confusion matirx
from sklearn.metrics import confusion_matrix

#import seaborn as sns
#from sklearn.metrics import roc_curve

####X_train
probas=resnet34_model.predict(X_train)
y_train_pred = np.argmax(probas,axis=1)

conf_mx = confusion_matrix(Y_train.astype(str), y_train_pred.astype(str))
conf_mx

def plot_confusion_matrix(matrix):
    #If you prefer color and a colorbar
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.title("resnet34_confusion_matrix",x=0.5,y=-0.12)
plt.savefig("./plot/resnet34_T_confusion_matrix.pdf", tight_layout=False)
plt.show()

###error analysis
#row： actual class， column：predicted classes
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.title("resnet34_confusion_matrix_errors",x=0.5,y=-0.12)
plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
plt.savefig("./plot/resnet34_T_confusion_matrix_errors.pdf", tight_layout=False)
plt.show()


####X_val
probas=resnet34_model.predict(X_val)
y_val_pred = np.argmax(probas,axis=1)

conf_mx = confusion_matrix(Y_val.astype(str), y_val_pred.astype(str))
conf_mx

def plot_confusion_matrix(matrix):
    #If you prefer color and a colorbar
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.title("resnet34_confusion_matrix",x=0.5,y=-0.12)
plt.savefig("./plot/resnet34_V_confusion_matrix.pdf", tight_layout=False)
plt.show()

###error analysis
#row： actual class， column：predicted classes
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.title("resnet34_confusion_matrix_errors",x=0.5,y=-0.12)
plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
plt.savefig("./plot/resnet34_V_confusion_matrix_errors.pdf", tight_layout=False)
plt.show()
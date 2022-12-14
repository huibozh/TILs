# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 19:08:22 2022

@author: Huibo Zhang
"""

import numpy as np 
import glob
import cv2
import os
import tensorflow as tf
#from tensorflow import keras
#from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense#, Dropout, Activation, Flatten, GlobalAveragePooling2D
import matplotlib.pyplot as plt

resnet50_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet50_model.add(pretrained_model)

resnet50_model.add(Flatten())
resnet50_model.add(Dense(512, activation='relu'))
resnet50_model.add(Dense(3, activation='softmax'))

resnet50_model.summary()


"""
###input data
"""
##### Training set:
#Read input images and assign labels based on folder names
print(os.listdir("training/train/"))

IMG_WIDTH = 180
IMG_HEIGHT = 180
IMG_CHANNELS = 3
SIZE = 180  #Resize images

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
SIZE = 180  #Resize images
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


####training
resnet50_model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy",metrics = ['accuracy'])
checkpointer = tf.keras.callbacks.ModelCheckpoint('resnet50_model_for_classification.h5', verbose=1, save_best_only=True)
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        checkpointer]

history = resnet50_model.fit(X_train,Y_train,
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
plt.savefig('./plot/resnet50_accuracy.pdf', dpi = 1000)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./plot/resnet50_loss.pdf', dpi = 1000)
plt.show()


"""
#######  confusion matirx   ############
"""
#####confusion matirx
from sklearn.metrics import confusion_matrix

#import seaborn as sns
#from sklearn.metrics import roc_curve

####X_train
probas=resnet50_model.predict(X_train)
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
plt.title("resnet50_confusion_matrix",x=0.5,y=-0.12)
plt.savefig("./plot/resnet50_T_confusion_matrix.pdf", tight_layout=False)
plt.show()

###error analysis
#row： actual class， column：predicted classes
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.title("resnet50_confusion_matrix_errors",x=0.5,y=-0.12)
plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
plt.savefig("./plot/resnet50_T_confusion_matrix_errors.pdf", tight_layout=False)
plt.show()


####X_val
probas=resnet50_model.predict(X_val)
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
plt.title("resnet50_confusion_matrix",x=0.5,y=-0.12)
plt.savefig("./plot/resnet50_V_confusion_matrix.pdf", tight_layout=False)
plt.show()

###error analysis
#row： actual class， column：predicted classes
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.title("resnet50_confusion_matrix_errors",x=0.5,y=-0.12)
plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
plt.savefig("./plot/resnet50_V_confusion_matrix_errors.pdf", tight_layout=False)
plt.show()
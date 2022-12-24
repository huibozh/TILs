# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 22:32:12 2022

@author: Admin
"""


import numpy as np 
import glob
import cv2
import os
#import tensorflow as tf
#from tensorflow import keras
# keras.applications import InceptionResNetV2
#from keras.models import Sequential
#from keras.layers import Flatten
#from keras.layers import Dense#, Dropout, Activation, GlobalAveragePooling2D
#import matplotlib.pyplot as plt
print(os.listdir("training/train/"))

IMG_WIDTH = 150
IMG_HEIGHT = 150
IMG_CHANNELS = 3
SIZE = 150  #Resize images

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
x_train = X_train.reshape(len(X_train),-1)



from sklearn.cluster import KMeans
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(x_train)
#Each instance was assigned to one of the 5 clusters:
y_pred

from pandas.core.frame import DataFrame
import pandas as pd

Y_pred = DataFrame(y_pred)
Y_pred.index.name="No"
Y_label = DataFrame(Y_train)
Y_label.index.name="No"
pred_results=pd.merge(Y_pred,Y_label,on='No')
pred_results.columns=["prediction","label"]
same = pred_results[pred_results["prediction"] == pred_results["label"]]
accuracy_value=same.shape[0]/pred_results.shape[0]
with open("accuracy_value.txt", 'w') as f: 
    print("accuracy_value:",accuracy_value, file = f)

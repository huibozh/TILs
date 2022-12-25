# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 15:58:55 2022

@author: Huibo Zhang
"""

import os
import numpy as np
import cv2
import glob
from keras import backend as K
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from pandas.core.frame import DataFrame
import pandas as pd

## autoencoder model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(152, 152, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 
model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


## input data
print(os.listdir("training/train/"))
IMG_WIDTH = 152
IMG_HEIGHT = 152
IMG_CHANNELS = 3
SIZE = 152  #Resize images

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


model.fit(X_train, X_train,epochs=50,shuffle=True)

#saving model
model.save('autoencoder_hbzh.h5')
#get encoder layer
get_encoded = K.function([model.layers[0].input], [model.layers[5].output])

X_encoded = get_encoded([X_train])[0]


###### Kmeans 
## data reshape
#method 1:
#X_encoded_reshape = X_encoded.reshape(X_encoded.shape[0], X_encoded.shape[1]*X_encoded.shape[2]*X_encoded.shape[3])    
#method 2:
X_encoded_reshape = X_encoded.reshape(len(X_encoded),-1)

kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X_encoded_reshape)


## get accuracy score 
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




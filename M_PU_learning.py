# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:47:35 2023

@author: Admin
"""

import os
import time

import sklearn
import numpy as np
import pandas as pd
from sklearn.utils import resample
from baggingPU import BaggingClassifierPU
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import glob
import cv2
from pandas.core.frame import DataFrame


### function
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]) + 4
    empty_cell = " " * columnwidth
    print("    " + empty_cell, end=' ')
    for label in labels:
        print("%{0}s".format(columnwidth) % 'pred_' + label, end=" ")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % 'true_' + label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if cell:
                print(cell, end=" ")
        print()


### input data
print(os.listdir("PU_learning/one_positive/")) # A_positive--Positive, B_positive,C_other--Negative
IMG_WIDTH = 150
IMG_HEIGHT = 150
IMG_CHANNELS = 3
SIZE = 150  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("PU_learning/one_positive/*"):
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
X_train = DataFrame(x_train)
X_train.index.name="No"
Y2_train = DataFrame(Y_train)
Y2_train.index.name="No"
Y2_train.columns=["label"]

data1= pd.merge(X_train,Y2_train,on='No')
print(data1.label.value_counts())
print('Has null values', data1.isnull().values.any())


"""
### replace some positive samples with negative samples
def random_undersampling(tmp_df, TARGET_LABEL):
    df_majority = tmp_df[tmp_df[TARGET_LABEL] == 0]
    df_minority = tmp_df[tmp_df[TARGET_LABEL] == 1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,              # sample without replacement
                                       n_samples=len(df_minority), # to match minority class
                                       random_state=None)        # reproducible results
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    print("Undersampling complete!")
    print(df_downsampled[TARGET_LABEL].value_counts())
    return df_downsampled

"""

df = data1.copy()

#Separate cols from label
NON_LBL = [c for c in df.columns if c != 'label']
X = df[NON_LBL]
y = df['label']

# Save the original labels and indices
y_orig = Y2_train
original_idx = np.where(data1.label == 1)

# imputing 15712 positives as negative, 2000 left
hidden_size = 15712
y.loc[
    np.random.choice(
        y[y == 1].index, 
        replace = False, 
        size = hidden_size
    )
] = 0

pd.Series(y).value_counts()
print('- %d samples and %d features' % (X.shape))
print('- %d positive out of %d total before hiding labels' % (sum(data1.label), len(data1.label)))
print('- %d positive out of %d total after hiding labels' % (sum(y), len(y)))


print('Training bagging classifier...')
pu_start = time.perf_counter()
bc = BaggingClassifierPU(RandomForestClassifier(n_estimators=100, random_state=42), 
                         n_estimators = 100, 
                         n_jobs = -1, 
                         max_samples = sum(y)  # Each training sample will be balanced 
                        )
bc.fit(X, y)
pu_end = time.perf_counter()
print('Done!')
print('Time:', pu_end - pu_start)

predict_result=bc.predict(X)
print('---- {} ----'.format('PU Bagging'))
print(print_cm(sklearn.metrics.confusion_matrix(y_orig, predict_result), labels=['Negative', 'Positive']))
print('')
print('Precision: ', precision_score(y_orig, predict_result))
print('Recall: ', recall_score(y_orig, predict_result))
print('Accuracy: ', accuracy_score(y_orig, predict_result))
print("Done!")

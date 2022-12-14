# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 00:57:36 2022

@author: Huibo Zhang
"""

###############predict new images#############################

import numpy as np 
import glob
import cv2
import os
from tensorflow import keras
from pandas.core.frame import DataFrame
import pandas as pd
#import matplotlib.pyplot as plt
#from functools import partial

## modify patch scale according to different models

vgg16_model=keras.models.load_model('vgg16_model_for_classification.h5')


print(os.listdir("prediction/"))
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
SIZE = 224  #Resize images
#Capture training data and labels into respective lists

for directory_path in glob.glob("prediction/*"):
    label = directory_path.split("\\")[-1]
    sample_path=directory_path
    pred_images = []
    pred_labels = [] 
    pred_id = []
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pred_images.append(img)
        pred_labels.append(label)
        pred_id.append(img_path)
    #Convert lists to arrays        
    X_pred = np.array(pred_images)
    X_pred = X_pred / 255.0
    #prediction
    y_proba = vgg16_model.predict(X_pred)
    #### calculating TIL score (TIL positive ratio)
    pred_ID=DataFrame(pred_id)
    Y_proba=DataFrame(y_proba)
    Y_proba.index.name='No'
    pred_ID.index.name='No'
    y_proba1=pd.merge(pred_ID,Y_proba,on='No')
    y_proba1.columns=['Patch_id','Positive','Negative','Other']
    y_proba1.to_csv(sample_path +"\\"+ label + "_predict_results.csv",index=False, header=True)

    TIL_positive=y_proba1[(y_proba1["Positive"] > y_proba1["Negative"])&(y_proba1["Positive"] > y_proba1["Other"])]
    TIL_positive.to_csv(sample_path +"\\" + "positive_patches.csv",index=False, header=True)
    
    TIL_negative=y_proba1[(y_proba1["Negative"] > y_proba1["Positive"])&(y_proba1["Negative"] > y_proba1["Other"])]
    TIL_negative.to_csv(sample_path +"\\" + "negative_patches.csv",index=False, header=True)
    
    #tumor_region=y_proba1[(y_proba1["Positive"] > y_proba1["Other"])&(y_proba1["Negative"] > y_proba1["Other"])]
    #row number ratio
    score=TIL_positive.shape[0]/(TIL_negative.shape[0] + TIL_positive.shape[0])
    #save predicted score 
    with open(sample_path +"\\"+ label + "_pred_score.txt", 'w') as f: 
        print(label, score, file = f)
    
    
###get prediction txt file
rootdir = r'./prediction'   
newfile = r'./prediction/TIL_scores.txt'  
paths = []   

for root, dirs, files in os.walk(rootdir):
    for file in files: 
        if file.endswith(".txt"):
            paths.append(os.path.join(root, file).encode('utf-8')) 
            
f = open(newfile,'w',encoding='utf-8')
for i in paths:
    for line in open(i,encoding='ISO-8859-1'):
        f.writelines(line)
        
f.close()
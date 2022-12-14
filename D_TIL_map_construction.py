# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:00:47 2022

@author: Huibo Zhang
"""
"""
delete all .svs files when getting prediction result directoies,
then run codes below:
"""
"""
#################  TIL map  ###################3
"""

import glob
import os
import matplotlib.pyplot as plt
#import numpy as np
#from numpy import array
import pandas as pd
#import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from matplotlib.pyplot import MultipleLocator


file_path=os.path.abspath(r"./prediction")
for directory_path in glob.glob("prediction/*"):
    label1 = directory_path.split("\\")[-1]
    #print(label1)
    #delete the first "for circle"(the top 2 lines) when there is only one directory
  ##### negative region
    X1= []
    Y1= []
    for img_path in glob.glob("prediction" +"\\" + label1 + "\\"+ "negative/*"):
        label2 = img_path.split("\\")[-1]
        #print(label2)
        
        str = label2.split('_');
        x1=label2.split('_')[0]
        y1=label2.split('_')[1]
        #print("X: "+x1)
        #print("Y: "+y1)
        X1.append(x1)
        x=pd.Series(X1)
        Y1.append(y1)
        y1=pd.Series(Y1)
    X1a=DataFrame(X1)
    Y1a=DataFrame(Y1)
    X1a.index.name='No'
    Y1a.index.name='No'
    data1=pd.merge(X1a,Y1a,on='No')
    data1.columns=['X','Y']
    data1['X'] = pd.to_numeric(data1['X'],errors='coerce')
    data1['Y'] = pd.to_numeric(data1['Y'],errors='coerce')

  ##### TIL region
    #file_path=os.path.abspath(r"./prediction")
    X2= []
    Y2= []
    for img_path2 in glob.glob("prediction" +"\\" + label1 + "\\"+ "positive/*"):
        label3 = img_path2.split("\\")[-1]
        #print(label3)
        str = label3.split('_');
        x2=label3.split('_')[0]
        y2=label3.split('_')[1]
        #print("X: "+x2)
        #print("Y: "+y2)
        X2.append(x2)
        x2=pd.Series(X2)
        Y2.append(y2)
        y2=pd.Series(Y2)
    X2a=DataFrame(X2)
    Y2a=DataFrame(Y2)
    X2a.index.name='No'
    Y2a.index.name='No'
    data2=pd.merge(X2a,Y2a,on='No')
    data2.columns=['X','Y']
    data2['X'] = pd.to_numeric(data2['X'],errors='coerce')
    data2['Y'] = pd.to_numeric(data2['Y'],errors='coerce')


    plt.figure(figsize=(20, 20))
## tumor region
    plt.scatter(x=data1.X,y=data1.Y,s=48, color='purple',marker='s',)
## TIL region
    plt.scatter(x=data2.X,y=data2.Y,s=48, color='red',marker='s',)
    plt.title("TIL map") 
    x_major_locator=MultipleLocator(1)
    y_major_locator=MultipleLocator(1)
    plt.gca().set_aspect(1)
#max value of axis
    plt.xlim(0,150)
    plt.ylim(0,150)
    plt.savefig(directory_path +"\\" + label1 + '_TIL.png', dpi = 1000)
    plt.savefig(directory_path +"\\" + label1 + '_TIL.pdf', dpi = 1000)
    plt.show()


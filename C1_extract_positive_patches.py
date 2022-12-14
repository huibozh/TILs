# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:53:21 2022

@author: Huibo Zhang
"""

import os
import pandas as pd
import shutil
import glob

"""
####### method1: extract positive patches from one directory

df = pd.read_csv(r'./prediction/fileA/positive_patches.csv')
file_name = df['Patch_id'].str[17:]# 17为测试文件所需字符串开始位置，到时候重新修改为实际的位置！！！
file_num = len(file_name)
# print(file_num)
file_path = r'./prediction/fileA/'
os.makedirs("prediction/fileA/positive")
save_path = r'./prediction/fileA/positive/'
filenames = os.listdir(file_path)

#read all lake file
for filename in filenames:
    old_dir = os.path.join(file_path,filename) 
    for i in range(file_num):
        if str(file_name[i]) in filename: 
            new_dir=os.path.join(save_path,filename)
            shutil.move(old_dir,new_dir) 
        else:
            continue
print('done!')
"""

"""
####### method2: extract positive patches from all directories
"""

for directory_path in glob.glob("prediction/*"):
    print(directory_path)
    df = pd.read_csv(directory_path +"\\" + "positive_patches.csv") #predict_results.csv 为测试文件，到时候改为positive.csv 文件
    file_name = df['Patch_id'].str[17:] # 17为测试文件所需字符串开始位置，到时候重新修改为实际的位置！！！
    file_num = len(file_name)
    # print(file_num)
    file_path = directory_path
    os.makedirs(directory_path + "\\" +"positive")
    save_path = directory_path + "\\" +"positive"
    filenames = os.listdir(file_path)
    #read all lake file
    for filename in filenames:
        old_dir = os.path.join(file_path,filename) 
        for i in range(file_num):
            if str(file_name[i]) in filename: 
                new_dir=os.path.join(save_path,filename)
                shutil.move(old_dir,new_dir) 
            else:
                continue
print('done!')


## extract negative patches from all directories
for directory_path in glob.glob("prediction/*"):
    print(directory_path)
    df = pd.read_csv(directory_path +"\\" + "negative_patches.csv") #predict_results.csv 为测试文件，到时候改为negative.csv 文件
    file_name = df['Patch_id'].str[17:] # 17为测试文件字符串位置，到时候重新修改为实际的位置！！！
    file_num = len(file_name)
    # print(file_num)
    file_path = directory_path
    os.makedirs(directory_path + "\\" +"negative")
    save_path = directory_path + "\\" +"negative"
    filenames = os.listdir(file_path)
    #read all lake file
    for filename in filenames:
        old_dir = os.path.join(file_path,filename) 
        for i in range(file_num):
            if str(file_name[i]) in filename: 
                new_dir=os.path.join(save_path,filename)
                shutil.move(old_dir,new_dir) 
            else:
                continue
print('done!')

"""
remove 03_other patches under directory_path 
"""

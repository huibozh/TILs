
"""
Created on Fri Oct 21 18:24:36 2022

author: Huibo Zhang
"""

import os

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
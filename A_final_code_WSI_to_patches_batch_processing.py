"""
Created on Sun Oct 16 00:57:36 2022

@author: Huibo Zhang
"""

#change environment C:\TUM\TCGA_pathomics\pkgs\openslide-win64\bin：
import openslide
#from openslide import open_slide
#from PIL import Image
import numpy as np
#from matplotlib import pyplot as plt
import tifffile as tiff
from openslide.deepzoom import DeepZoomGenerator
import os
import glob

#Load a level image, normalize the image and digitally extract H and E images
#As described in video 122: https://www.youtube.com/watch?v=yUrwEYgZUsA

#change environment：C:\TUM\TCGA_pathomics\pkgs：
from normalize_HnE import norm_HnE

file_path=os.path.abspath(r"./example")
for directory_path in glob.glob("example/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    file_name=file_path +"\\" + label[0: 23]
    norm_tile_dir_name = file_name
    os.makedirs(file_name)
    slide = openslide.OpenSlide(file_path+"\\" + label)
    objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    #print("The objective power is: ", objective)
    tiles = DeepZoomGenerator(slide, tile_size=150, overlap=0, limit_bounds=False)
    #print("The number of levels in the tiles object are: ", tiles.level_count)
    #print("The dimensions of data in each level are: ", tiles.level_dimensions)
    #Total number of tiles in the tiles object
    #print("Total number of tiles = : ", tiles.tile_count)

    if objective < 40.0:
        level= tiles.level_count -1
        cols, rows = tiles.level_tiles[level]
    else:
        level=tiles.level_count -2
        cols, rows = tiles.level_tiles[level]
    for row in range(rows):
        for col in range(cols):
            try:
              tile_name = str(col) + "_" + str(row)
              temp_tile = tiles.get_tile(level, (col, row))
              temp_tile_RGB = temp_tile.convert('RGB')
              temp_tile_np = np.array(temp_tile_RGB)
              if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
                  print("Processing tile number:", tile_name)
                  norm_img, H_img, E_img = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)
              #Save the norm tile   
                  tiff.imsave(norm_tile_dir_name +"\\" + tile_name + "_" + label[0: 23] + "_norm.tif", norm_img)
              else:
                print("NOT PROCESSING TILE:", tile_name)
            except:
                pass
            continue       
    
        
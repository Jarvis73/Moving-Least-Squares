#! /usr/bin/env python27
# -*- coding: utf-8 -*-

"""
Load tif data and save it into a *.npz file

@author: Jarvis ZHANG
@date: 2017/8/7
@editor: VS Code
"""

import os
import platform
import numpy as np
from libtiff import TIFF

IMG_SIZE = 512
PAD_SIZE = 30   # 572 = 512 + 30 + 30
IMG_SLICES = 30

linux_data_dir = "/home/jarvis/DataSet/ISBI Challenge"
win_data_dir = "E:\\DataSet\\ISBI Challenge"
if "Windows" in platform.system():
    data_dir = win_data_dir
elif "Linux" in platform.system():
    data_dir = linux_data_dir

train_images_path = os.path.join(data_dir, "train-volume.tif")
train_labels_path = os.path.join(data_dir, "train-labels.tif")
test_images_path = os.path.join(data_dir, "test-volume.tif")

def _read_data(path):
    ''' Read data from given tiff files.  
    Libtiff is only supported by python 2.7
    ### Params:
        * path - string: path of the data
    ### Return:
        * dataset - ndarray: images or labels
    '''
    container = np.empty((IMG_SLICES, IMG_SIZE + PAD_SIZE * 2, IMG_SIZE + PAD_SIZE * 2))
    # load data
    tif = TIFF.open(path)
    for i, image in enumerate(tif.iter_images()):
        container[i] = np.pad(image, pad_width=((PAD_SIZE,)*2,)*2, mode='symmetric')
    tif.close()
    return container

# Read data from .tif files and store as ndarray
train_images = _read_data(train_images_path)
train_labels = _read_data(train_labels_path)
test_images = _read_data(test_images_path)
np.savez_compressed(
    os.path.join(data_dir, "all_img_lab"), 
    train_images=train_images, 
    train_labels=train_labels, 
    test_images=test_images
)


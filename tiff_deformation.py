#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import matplotlib.pyplot as plt
from img_utils import (mls_affine_deformation_inv, 
                       mls_rigid_deformation_inv, 
                       mls_similarity_deformation_inv)

all_img_lab = "all_img_lab.npz"
linux_data_dir = "/home/jarvis/DataSet/ISBI Challenge"
all_img_lab_path = os.path.join(linux_data_dir, all_img_lab)


def generate_ctrl_pts(height, width, size=5, range=5):
    ''' Generate control points p and q. 
        p: size x size grids
        q: size x size grids with noise
        noise: from uniform distribution

    ### Params:
        * height - integer: height of the image
        * width - integer: width of the image
        * size - integer: grid size
        * range - bound of the uniform dirtribution, [-range, range]
    ### Return:
        A tuple of p and q
    '''
    x, y = np.meshgrid(np.linspace(0, height, size), np.linspace(0, width, size))
    p = np.concatenate((x.reshape(size**2, 1), y.reshape(size**2, 1)), axis=1)
    noise = np.random.uniform(-range, range, (size, size, 2))
    noise[[0,-1], :, :] = 0
    noise[:, [0,-1], :] = 0
    q = p + noise.reshape(size**2, 2)
    return p, q


if __name__ == "__main__":
    npzfile = np.load(all_img_lab_path)

    train_images = npzfile['train_images']

    height = train_images.shape[1]
    width = train_images.shape[2]

    plt.subplot(221)
    plt.imshow(train_images[0], cmap='gray')
    plt.title("Original image")

    for i in range(3):
        p, q = generate_ctrl_pts(height, width, size=5, range=20)
        rigid = mls_rigid_deformation_inv(train_images[0], p, q)
        plt.subplot(2, 2, i+2)
        plt.imshow(rigid, cmap='gray')
        plt.title("Generate rigid " + str(i + 1))

    plt.show()


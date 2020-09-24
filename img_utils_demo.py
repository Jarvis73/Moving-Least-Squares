#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@update: 2020/9/25
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from img_utils import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation


def show_example():
    img = plt.imread(os.path.join(sys.path[0], "double.jpg"))
    plt.imshow(img)
    plt.show()

def demo(fun, name):
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    image = plt.imread(os.path.join(sys.path[0], "mr_big_ori.jpg"))

    plt.figure(figsize=(8, 3))
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(image)
    plt.title("Original Image")
    
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.title("%s Deformation \n Sampling density 1"%name)
    transformed_image = fun(image, p, q, alpha=1, density=0.7)
    plt.subplot(133)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.title("%s Deformation \n Sampling density 0.7"%name)

    plt.tight_layout(w_pad=0.1)
    plt.show()

def demo2(fun, name):
    ''' 
        Smiled Monalisa  
    '''
    
    p = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
    ])
    q = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [207, 238], [261, 237], [199, 253], [232, 281], [279, 249]
    ])
    image = plt.imread(os.path.join(sys.path[0], "monalisa.jpg"))
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(image)
    plt.title("Original Image")
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(transformed_image)
    plt.title("%s Deformation" % name)
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    demo(mls_affine_deformation, "Affine")
    demo(mls_similarity_deformation, "Similarity")
    demo(mls_rigid_deformation, "Rigid")

    demo2(mls_affine_deformation, "Affine")
    demo2(mls_similarity_deformation, "Similarity")
    demo2(mls_rigid_deformation, "Rigid")

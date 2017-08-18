#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimSun'
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)


def show_example():
    img = plt.imread(os.path.join(sys.path[0], "double.jpg"))
    plt.imshow(img)
    plt.show()

def demo(fun, fun_inv, name):
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    image = plt.imread(os.path.join(sys.path[0], "mr_big_ori.jpg"))
    plt.subplot(231)
    plt.imshow(image)
    plt.title("原图")
    if fun is not None:
        transformed_image = fun(image, p, q, alpha=1, density=1)
        plt.subplot(232)
        plt.imshow(transformed_image)
        plt.title("%s变换 -- 采样密度 1"%name)
        transformed_image = fun(image, p, q, alpha=1, density=0.7)
        plt.subplot(235)
        plt.imshow(transformed_image)
        plt.title("%s变换 -- 采样密度 0.7"%name)
    if fun_inv is not None:
        transformed_image = fun_inv(image, p, q, alpha=1, density=1)
        plt.subplot(233)
        plt.imshow(transformed_image)
        plt.title("逆%s变换 -- 采样密度 1"%name)
        transformed_image = fun_inv(image, p, q, alpha=1, density=0.7)
        plt.subplot(236)
        plt.imshow(transformed_image)
        plt.title("逆%s变换 -- 采样密度 0.7"%name)

    plt.show()

def demo2(fun):
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
    plt.subplot(121)
    plt.imshow(image)
    transformed_image = fun(image, p, q, alpha=1, density=1)
    plt.subplot(122)
    plt.imshow(transformed_image)
    plt.show()


if __name__ == "__main__":
    #affine deformation
    #demo(mls_affine_deformation, mls_affine_deformation_inv, "仿射")
    #demo2(mls_affine_deformation_inv)

    #similarity deformation
    #demo(mls_similarity_deformation, mls_similarity_deformation_inv, "相似")
    #demo2(mls_similarity_deformation_inv)

    #rigid deformation
    #demo(mls_rigid_deformation, mls_rigid_deformation_inv, "刚性")
    demo2(mls_rigid_deformation_inv)

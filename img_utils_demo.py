#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jian-Wei ZHANG
@date: 2017/8/8
@update: 2020/9/25
@update: 2021/7/14: simplify usage
"""

import numpy as np
import matplotlib.pyplot as plt
from img_utils import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation
from PIL import Image


def demo():
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    
    image = np.array(Image.open("images/toy.jpg"))
    
    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    
    affine = mls_affine_deformation(vy, vx, p, q, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(affine)]

    similar = mls_similarity_deformation(vy, vx, p, q, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(similar)]

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug3 = np.ones_like(image)
    aug3[vx, vy] = image[tuple(rigid)]


    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")    
    ax[1].imshow(aug1)
    ax[1].set_title("Affine Deformation")
    ax[2].imshow(aug2)
    ax[2].set_title("Similarity Deformation")
    ax[3].imshow(aug3)
    ax[3].set_title("Rigid Deformation")

    for x in ax.flat:
        x.axis("off")

    plt.tight_layout(w_pad=0.1)
    plt.show()


def demo2():
    """ Smiled Monalisa """
    
    p = np.array([[0, 0], [0, 517], [798, 0], [798, 517],
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
    ])
    q = np.array([[0, 0], [0, 517], [798, 0], [798, 517],
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [207, 238], [261, 237], [199, 253], [232, 281], [279, 249]
    ])

    image = np.array(Image.open("images/monalisa.jpg"))

    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    rigid = mls_rigid_deformation(vy, vx, p, q, alpha=1)
    aug = np.ones_like(image)
    aug[vx, vy] = image[tuple(rigid)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug)
    ax[1].set_title("Rigid Deformation")

    for x in ax.flat:
        x.axis("off")
    
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    # demo()
    demo2()


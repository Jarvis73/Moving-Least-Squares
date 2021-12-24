#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jian-Wei ZHANG
@email: zjw.cs@zju.edu.cn
@date: 2017/8/8
@update: 2020/9/25
@update: 2021/7/14: Simplify usage
@update: 2021/12/24: Fix bugs and add an example of random control points (see `demo2()`)
"""

import numpy as np
import matplotlib.pyplot as plt
from img_utils import mls_affine_deformation, mls_similarity_deformation, mls_rigid_deformation
from PIL import Image


def demo():
    p = np.array([
        [155, 30], [155, 125], [155, 225],
        [235, 100], [235, 160], [295, 85], [293, 180]
    ])
    q = np.array([
        [211, 42], [155, 125], [100, 235],
        [235, 80], [235, 140], [295, 85], [295, 180]
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
    np.random.seed(1234)
    
    image = np.array(Image.open("images/monalisa.jpg"))
    height, width, _ = image.shape
    
    # Define deformation grid
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    # ================ Control points group 1 (manually specified) ==================
    p1 = np.array([[0, 0], [517, 0], [0, 798], [517, 798],
        [140, 186], [135, 295], [181, 208], [181, 261], [203, 184], [202, 304], [225, 213], 
        [225, 243], [244, 211], [244, 253], [254, 195], [281, 232], [252, 285]
    ])
    q1 = np.array([[0, 0], [517, 0], [0, 798], [517, 798],
        [140, 186], [135, 295], [181, 208], [181, 261], [203, 184], [202, 304], [225, 213], 
        [225, 243], [238, 207], [237, 261], [253, 199], [281, 232], [249, 279]
    ])

    rigid1 = mls_rigid_deformation(vy, vx, p1, q1, alpha=1)
    aug1 = np.ones_like(image)
    aug1[vx, vy] = image[tuple(rigid1)]

    # ====================== Control points group 1 (random) =======================
    p2 = np.stack((
        np.random.randint(0, height, size=13), 
        np.random.randint(0, width, size=13),
    ), axis=1)
    q2 = p2 + np.random.randint(-20, 20, size=p2.shape)

    rigid2 = mls_rigid_deformation(vy, vx, p2, q2, alpha=1)
    aug2 = np.ones_like(image)
    aug2[vx, vy] = image[tuple(rigid2)]

    fig, ax = plt.subplots(1, 3, figsize=(13, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(aug1)
    ax[1].set_title("Manually specified control points")
    ax[2].imshow(aug2)
    ax[2].set_title("Random control points")

    for x in ax.flat:
        x.axis("off")
    
    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


def read_tif(frame):
    image_pil = Image.open("images/train-volume.tif")
    image_pil.seek(frame)
    image = np.array(image_pil)
    label_pil = Image.open("images/train-labels.tif")
    label_pil.seek(frame)
    label = np.array(label_pil)

    return image, label


def demo3():
    image, label = read_tif(1)
    image = np.pad(image, ((30, 30), (30, 30)), mode='symmetric')
    label = np.pad(label, ((30, 30), (30, 30)), mode='symmetric')

    height, width = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)

    def augment(p, q, mode='affine'):
        if mode.lower() == 'affine':
            transform = mls_affine_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == 'similar':
            transform = mls_similarity_deformation(vy, vx, p, q, alpha=1)
        elif mode.lower() == 'rigid':
            transform = mls_rigid_deformation(vy, vx, p, q, alpha=1)
        else:
            raise ValueError

        aug_img = np.ones_like(image)
        aug_img[vx, vy] = image[tuple(transform)]
        aug_lab = np.ones_like(label)
        aug_lab[vx, vy] = label[tuple(transform)]

        return aug_img, aug_lab

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[1, 0].imshow(label, cmap='gray')
    ax[1, 0].set_title("Original Label")
    
    np.random.seed(1234)
    p = np.c_[np.random.randint(0, height, size=32), np.random.randint(0, width, size=32)]
    q = p + np.random.randint(-15, 15, size=p.shape)
    q[:, 0] = np.clip(q[:, 0], 0, height)
    q[:, 1] = np.clip(q[:, 1], 0, width)
    p = np.r_[p, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])]  # fix corner points
    q = np.r_[q, np.array([[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]])]  # fix corner points

    for i, mode in enumerate(['Affine', 'Similar', 'Rigid']):
        aug_img, aug_lab = augment(p, q, mode)
        ax[0, i + 1].imshow(aug_img, cmap='gray')
        ax[0, i + 1].set_title(f"{mode} Deformated Image")
        ax[1, i + 1].imshow(aug_lab, cmap='gray')
        ax[1, i + 1].set_title(f"{mode} Deformated Label")

    for x in ax.flat:
        x.axis('off')

    plt.tight_layout(w_pad=1.0, h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    demo()
    # demo2()
    # demo3()


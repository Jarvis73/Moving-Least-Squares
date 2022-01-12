# Moving Least Squares (MLS) (Numpy & PyTorch)

## Introduction
**Moving least squares** is a method of reconstructing continuous functions from a set of unorganized point samples via the calculation of a weighted least squares measure biased towards the region around the point at which the reconstructed value is requested.

In computer graphics, the moving least squares method is useful for reconstructing a surface from a set of points. Often it is used to create a 3D surface from a point cloud through either downsampling or upsampling.

## Methods
* Affine deformation
* Similarity deformation
* Rigid deformation

Please check the `demo.py` for usage.

## Results

* Toy

![Deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/toy_results.png)

* Monalisa (Rigid)

![Rigid deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/monalisa_rigid.png)

* Cells ([Download data](https://github.com/alexklibisz/isbi-2012/tree/master/data))

![Rigid Deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/cell_deformation.png)

The original label is overlapped on the deformed labels for better comparison.

![Rigid Deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/cell_deformation_with_alpha.png)

## Code list
* `img_utils.py`: Numpy implementation of the algorithms
* `img_utils_pytorch.py`: PyTorch implementation of the algorithms
* `interp_torch.py`: 
* `demo.py`: Demo programs

### Optimize memory usage

*   Here lists some examples of memory usage and running time of the numpy implementation

| Image Size  | Control Points | Affine         | Similarity     | Rigid          |
| ----------- | -------------- | -------------- | -------------- | -------------- |
| 500 x 500   | 16             | 0.57s / 0.15GB | 0.99s / 0.16GB | 0.89s / 0.13GB |
| 500 x 500   | 64             | 1.6s / 0.34GB  | 3.7s / 0.3GB   | 3.6s / 0.2GB   |
| 1000 x 1000 | 64             | 7.7s / 1.1GB   | 17s / 0.98GB   | 15s / 0.82GB   |
| 2000 x 2000 | 64             | 30s / 4.2GB    | 65s / 3.6GB    | 69s / 3.1GB    |

*   Estimate memory usage for large image: (h x w x N x 4 x 2) x 2~2.5
    *   h, w: image size
    *   N: number of control points
    *   4: float32
    *   2: coordinates (x, y)
    *   2~2.5: intermediate results


### Accelerating by PyTorch

The algorithm is also implemented with [PyTorch](https://pytorch.org/) and has faster speed benefiting from the CUDA acceleration.

* Rigid deformation

| Image Size  | Control Points | Numpy     | PyTorch with CUDA |
| ----------- | -------------- | --------- | -------- |
| 100 x 100   | 16             | 0.025s    | 0.128s   |
| 500 x 500   | 16             | 0.753s    | 0.187s   |
| 500 x 500   | 32             | 1.934s    | 0.205s   |
| 500 x 500   | 64             | 3.384s    | 0.483s   |
| 1000 x 1000 | 64             | 13.089s   | 0.663s   |
| 2000 x 2000 | 64             | 61.874s   | 1.784s   |

(* Tested on pytorch=1.6.0 with cudatoolkit=10.1)

### Update

*   **2022-01-12**   Implement three algorithms with PyTorch

*   **2021-12-24:**  Fix a bug of nan values in `mls_rigid_deformation()`. (see issue #13)

*   **2021-07-14:**  Optimize memory usage. Now a 2000x2000 image with 64 control points spend about 4.2GB memory. (20GB in the previous version)

*   **2020-09-25:**  No need for so-called inverse transformation. Just transform target pixels to the corresponding source pixels.


## Reference

[1] Schaefer S, Mcphail T, Warren J. Image deformation using moving least squares[C]// ACM SIGGRAPH. ACM, 2006:533-540.

[2] `interp` implementation in `interp_torch.py`. [Github: aliutkus/torchinterp1d](https://github.com/aliutkus/torchinterp1d)

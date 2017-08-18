# Moving Least Squares (MLS)

## Introduction
**Moving least squares** is a method of reconstructing continuous functions from a set of unorganized point samples via the calculation of a weighted least squares measure biased towards the region around the point at which the reconstructed value is requested.

In computer graphics, the moving least squares method is useful for reconstructing a surface from a set of points. Often it is used to create a 3D surface from a point cloud through either downsampling or upsampling.

## Methods
* Affine deformation (inverse)
* Similarity deformation (inverse)
* Rigid deformation (inverse)

## Preview
* Toy

![Affine deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/toy_1_affine.png)

![Similarity deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/toy_2_similarity.png)

![Rigid deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/toy_3_rigid.png)

* Monalisa

![Rigid deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/monalisa_3_rigid.png)

* Cells

![Rigid Deformation](https://github.com/jarvis73/Moving-Least-Squares/raw/master/images/tiff_deformation.png)

## Code list
* `img_utils.py`: Implementation of the algorithms
* `img_utils_demo.py`: Demo program
* `read_tif.py`: TIF file reader
* `tiff_deformation.py`: Demo program

## Reference
[1] Schaefer S, Mcphail T, Warren J. Image deformation using moving least squares[C]// ACM SIGGRAPH. ACM, 2006:533-540.

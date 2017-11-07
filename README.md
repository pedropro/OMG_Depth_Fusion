# Optimal Mixture of Gaussians for Depth Fusion
Implementation of the filter proposed in:
Probabilistic RGB-D Odometry based on Points, Lines and Planes Under Depth Uncertainty, P. Proença and Y. Gao https://arxiv.org/abs/1706.04034

The OMG filter is aimed at denoising and hole-filling the depth maps given by a depth sensor, but also (more importantly) capturing the depth uncertainty though spatio-temporal observations, which proved to be useful as an input to a probabilistic visual odometry system.

The code runs the OMG on RGB-D dataset sequences (e.g. ICL_NUIM and TUM). Some examples are included in this repository.
This release does not include the VO system, thus instead of using the VO frame-to-frame pose, we rely on the groundtruth pose to transform the old measurements (the registered point cloud) to the current frame.

# Dependencies

* Opencv
* Eigen3

# Ubuntu Instructions
Tested with Ubuntu 14.04

To compile, inside the directory ``./OMG`` type:
```
mkdir build
cd build
cmake ../
make
```

Use one of this to run the executable:
```./test_OMG 4 9 TUM_RGBD sitting_static_short```
```./test_OMG 4 9 ICL_NUIM living_room_traj0_frei_png_short```

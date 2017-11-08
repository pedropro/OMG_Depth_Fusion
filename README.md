# Optimal Mixture of Gaussians for Depth Fusion
Implementation of the smoother proposed in:

**Paper:** [Probabilistic RGB-D Odometry based on Points, Lines and Planes Under Depth Uncertainty](https://arxiv.org/abs/1706.04034), P. Proença and Y. Gao

The OMG filter is aimed at denoising and hole-filling the depth maps given by a depth sensor, but also (more importantly) capturing the depth uncertainty though spatio-temporal observations, which proved to be useful as an input to the probabilistic visual odometry system proposed in the related paper.

The code runs the OMG on RGB-D dataset sequences (e.g. ICL_NUIM and TUM). Some examples are included in this repository.
This is a stripped down version: It does not include the VO system, thus instead of using the VO frame-to-frame pose, we rely on the groundtruth poses to transform the old measurements (the registered point cloud) to the current frame. Also, the pose uncertainty condition was removed. The sensor model employed is specific for Kinect v1. Thus, to use with other sensors (e.g. ToF cameras), this should be changed.

## Dependencies

* OpenCV
* Eigen3

## Ubuntu Instructions
Tested with Ubuntu 14.04

To compile, inside the directory ``./OMG`` type:
```
mkdir build
cd build
cmake ../
make
```
To run the executable type:

```./test_OMG 4 9 TUM_RGBD sitting_static_short```

or

```./test_OMG 4 9 ICL_NUIM living_room_traj0_frei_png_short```

## Windows Instructions

Tested configuration: Windows 8.1 with Visual Studio 10 & 12

This version includes already a VC11 project.
Just make the necessary changes to link the project with OpenCV and Eigen.

## Usage

General Format
```./test_OMG <max_nr_frames> <consistency_threshold> <dataset_directory> <sequence_name>```

* ***max_nr_frames:*** is the window size - 1, i.e., the maximum number of previous frames used for fusion
* ***consistency threshold:*** is used to avoid fusing inconsistent measurements, if the squared distance between a new measurement and the current estimate is more than current uncertainty times this threshold than the new measurement is ignored. Setting this higher may produce better quality but will capture less the temporal uncertainty

## Data

Two short RGB-D sequences are included as examples. To add more sequences, download from:
[TUM_RGBD](https://vision.in.tum.de/data/datasets/rgbd-dataset) or
[ICL-NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)
and place them inside the directory ```./Data``` the same way it has been done for the examples.
Then add the respective camera parameters in the same format as the examples as ```calib_params.xml```

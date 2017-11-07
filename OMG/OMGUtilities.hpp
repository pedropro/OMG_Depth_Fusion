/*
*  Copyright 2017 Pedro Proença <p.proenca@surrey.ac.uk> (University of Surrey)
*/
/*
 *  Copyright 2017 Pedro Proenca <p.proenca@surrey.ac.uk> (University of Surrey)
 */

#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <math.h>

using namespace std;

int nr_colors = 64;
static float colormap_jet [192] = 
{ 0, 0, 0.5625
, 0, 0, 0.6250
, 0, 0, 0.6875
, 0, 0, 0.7500
, 0, 0, 0.8125
, 0, 0, 0.8750
, 0, 0, 0.9375
, 0, 0, 1.0000
, 0, 0.0625, 1.0000
, 0, 0.1250, 1.0000
, 0, 0.1875, 1.0000
, 0, 0.2500, 1.0000
, 0, 0.3125, 1.0000
, 0, 0.3750, 1.0000
, 0, 0.4375, 1.0000
, 0, 0.5000, 1.0000
, 0, 0.5625, 1.0000
, 0, 0.6250, 1.0000
, 0, 0.6875, 1.0000
, 0, 0.7500, 1.0000
, 0, 0.8125, 1.0000
, 0, 0.8750, 1.0000
, 0, 0.9375, 1.0000
, 0, 1.0000, 1.0000
, 0.0625, 1.0000, 0.9375
, 0.1250, 1.0000, 0.8750
, 0.1875, 1.0000, 0.8125
, 0.2500, 1.0000, 0.7500
, 0.3125, 1.0000, 0.6875
, 0.3750, 1.0000, 0.6250
, 0.4375, 1.0000, 0.5625
, 0.5000, 1.0000, 0.5000
, 0.5625, 1.0000, 0.4375
, 0.6250, 1.0000, 0.3750
, 0.6875, 1.0000, 0.3125
, 0.7500, 1.0000, 0.2500
, 0.8125, 1.0000, 0.1875
, 0.8750, 1.0000, 0.1250
, 0.9375, 1.0000, 0.0625
, 1.0000, 1.0000, 0
, 1.0000, 0.9375, 0
, 1.0000, 0.8750, 0
, 1.0000, 0.8125, 0
, 1.0000, 0.7500, 0
, 1.0000, 0.6875, 0
, 1.0000, 0.6250, 0
, 1.0000, 0.5625, 0
, 1.0000, 0.5000, 0
, 1.0000, 0.4375, 0
, 1.0000, 0.3750, 0
, 1.0000, 0.3125, 0
, 1.0000, 0.2500, 0
, 1.0000, 0.1875, 0
, 1.0000, 0.1250, 0
, 1.0000, 0.0625, 0
, 1.0000, 0, 0
, 0.9375, 0, 0
, 0.8750, 0, 0
, 0.8125, 0, 0
, 0.7500, 0, 0
, 0.6875, 0, 0
, 0.6250, 0, 0
, 0.5625, 0, 0
, 0.5000, 0, 0};


namespace utils{
	bool loadCalibParameters(string filepath, cv:: Mat & intrinsics_rgb, cv::Mat & dist_coeffs_rgb, cv:: Mat & intrinsics_ir, cv::Mat & dist_coeffs_ir, cv::Mat & R, cv::Mat & T){

		cv::FileStorage fs(filepath,cv::FileStorage::READ);
		if (fs.isOpened()){
			fs["RGB_intrinsic_params"]>>intrinsics_rgb;
			fs["RGB_distortion_coefficients"]>>dist_coeffs_rgb;
			fs["IR_intrinsic_params"]>>intrinsics_ir;
			fs["IR_distortion_coefficients"]>>dist_coeffs_ir;
			fs["Rotation"]>>R;
			fs["Translation"]>>T;
			fs.release();
			return true;
		}else{
			cerr<<"Calibration file missing"<<endl;
			return false;
		}
	}

	int remove_invalid_points(cv::Mat & X_map, cv::Mat & Y_map, cv::Mat & Z_map, cv::Mat & Range_map, cv::Mat & Var_map, float * point_cloud_array, float * range_array, float * var_array){
		int width = Z_map.cols;
		int height = Z_map.rows;
		int mxn =height*width;

		float * sX = (float*)(X_map.data);
		float * sY = (float*)(Y_map.data);
		float * sZ = (float*)(Z_map.data);
		float * sV = (float*)(Var_map.data);
		float * sR = (float*)(Range_map.data);
		float * dX = point_cloud_array;
		float * dY = &point_cloud_array[mxn];
		float * dZ = &point_cloud_array[2*mxn];
		float * dR = range_array;
		float * dVar = var_array;
		int nr_valid_pts=0; 
		for(int v=0; v< mxn; v++){
			float Z = *sZ++;
			if (Z>0){
				*dX++ = *sX;
				*dY++ = *sY;
				*dZ++ = Z;
				*dVar++ = *sV;
				*dR++ = *sR;
				nr_valid_pts++;
			}
			sX++; sY++; sV++; sR++;
		}
		return nr_valid_pts;
	}

	void convertQuaternion2SO3(double * q, Eigen::Matrix3d & R){
		// Assumes right-handed quaternion
		// TODO reuse terms
		R(0,0) = q[3]*q[3] + q[0]*q[0] - q[1]*q[1] - q[2]*q[2]; 
		R(0,1) = 2*(q[0]*q[1]-q[3]*q[2]);
		R(0,2) = 2*(q[0]*q[2]+q[3]*q[1]);
		R(1,0) = 2*(q[0]*q[1]+q[3]*q[2]);
		R(1,1) = q[3]*q[3] - q[0]*q[0] + q[1]*q[1] - q[2]*q[2];
		R(1,2) = 2*(q[1]*q[2]-q[3]*q[0]);
		R(2,0) = 2*(q[0]*q[2]-q[3]*q[1]);
		R(2,1) = 2*(q[1]*q[2]+q[3]*q[0]);
		R(2,2) = q[3]*q[3] - q[0]*q[0] - q[1]*q[1] + q[2]*q[2];

	}

	void convertGray2jet(cv::Mat & source, cv::Mat_<cv::Vec3f> & destination, float z_min = 500, float z_max = 5000){

		float range = z_max-z_min;

		float* o_px;
		float* d_px;

		for(int i=0; i< source.rows; i++){

			o_px = source.ptr<float>(i);
			d_px = destination.ptr<float>(i);
			for(int j=0; j< source.cols; j++){
				float z = *o_px++;
				if (z>0){
					int clr = (min(z,z_max)-z_min)*(nr_colors-1)/range;
					d_px[j*3] = colormap_jet[clr*3];
					d_px[j*3+1] = colormap_jet[clr*3+1];
					d_px[j*3+2] = colormap_jet[clr*3+2];
				}else{
					d_px[j*3] = 0.0f;
					d_px[j*3+1] = 0.0f;
					d_px[j*3+2] = 0.0f;
				}
			}
		}

	}

}

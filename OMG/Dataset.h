/*
 *  Copyright 2017 Pedro Proenca <p.proenca@surrey.ac.uk> (University of Surrey)
 */

#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <math.h> 

using namespace std;

class Dataset
{
public:
	enum Type {Real, Synthetic} dataset_type;
	vector<string> rgb_files;
	vector<string> d_files;
	vector<double> rgb_tstamps;
	vector<double> d_tstamps;
	vector<bool> has_gt_pose;
	vector<double*> gt_poses;
	Dataset(string filename, string dataset_name);
	void load_and_align_pose_gt(string filename);
};

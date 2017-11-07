/*
*  Copyright 2017 Pedro Proença <p.proenca@surrey.ac.uk> (University of Surrey)
*/

#include "Dataset.h"

Dataset::Dataset(string filepath, string dataset_name){

	if (dataset_name.compare("TUM_RGBD")==0){
		this->dataset_type = Real;
	}else{
		this->dataset_type = Synthetic;
	}

	FILE * pFile;
	vector<string> rgb_files_tmp;
	vector<string> d_files_tmp;
	vector<double> rgb_tstamps_tmp;
	vector<double> d_tstamps_tmp;

	// Read rgb list
	stringstream rgb_list_file;

	if (dataset_type == Synthetic){

		rgb_list_file<<filepath<<"/associations.txt";
		const string tmp = rgb_list_file.str();
		const char* cstr = tmp.c_str();
		pFile = fopen(cstr,"r");
		char buff[500];
		while(fgets(buff,500,pFile)!=NULL){
			char * head  = strtok(buff," ");
			char * tail  = strtok(NULL," ");
			double tstamp = strtod(head, NULL);
			string depth_frame_name(tail);
			double rgb_tstamp = strtod(head, NULL);
			strtok(NULL," ");
			tail  = strtok(NULL,"\n");
			string rgb_frame_name(tail);

			stringstream d_frame_path,rgb_frame_path;
			d_frame_path<<filepath<<"/"<<depth_frame_name;
			rgb_frame_path<<filepath<<"/"<<rgb_frame_name;

			rgb_files.push_back(rgb_frame_path.str());
			d_files.push_back(d_frame_path.str());
			rgb_tstamps.push_back(tstamp);
			d_tstamps.push_back(tstamp);
		}
	}else{
		rgb_list_file<<filepath<<"/rgb.txt";

		const string tmp = rgb_list_file.str();
		const char* cstr = tmp.c_str();
		pFile = fopen(cstr,"r");
		if(pFile){
			char buff[500];

			//Skip first 3 lines
			fgets(buff,500,pFile);
			fgets(buff,500,pFile);
			fgets(buff,500,pFile);

			while(fgets(buff,500,pFile)!=NULL){

				char * head  = strtok(buff," ");
				char * tail  = strtok(NULL," \n");
				double rgb_tstamp = strtod(head, NULL);
				string rgb_frame_name(tail);
				stringstream rgb_frame_path;
				rgb_frame_path<<filepath<<"/"<<rgb_frame_name;
				rgb_files_tmp.push_back(rgb_frame_path.str());
				rgb_tstamps_tmp.push_back(rgb_tstamp);
			}
			fclose(pFile);
		}
		else{
			cerr<<"Dataset does not exist"<<endl;
		}

		// Read depth list
		stringstream d_list_file;
		d_list_file<<filepath<<"/depth.txt";

		const string tmp_2= d_list_file.str();
		cstr = tmp_2.c_str();
		pFile = fopen(cstr,"r");
		if(pFile){
			char buff[500];

			//Skip first 3 lines
			fgets(buff,500,pFile);
			fgets(buff,500,pFile);
			fgets(buff,500,pFile);

			while(fgets(buff,500,pFile)!=NULL){

				char * head  = strtok(buff," ");
				char * tail  = strtok(NULL," \n");
				double d_tstamp = strtod(head, NULL);
				string d_frame_name(tail);
				stringstream d_frame_path;
				d_frame_path<<filepath<<"/"<<d_frame_name;
				d_files_tmp.push_back(d_frame_path.str());
				d_tstamps_tmp.push_back(d_tstamp);
			}
			fclose(pFile);
		}
		else{
			cerr<<"Dataset does not exist"<<endl;
		}

		sort(rgb_tstamps_tmp.begin(), rgb_tstamps_tmp.end());
		sort(d_tstamps_tmp.begin(), d_tstamps_tmp.end());

		for(int i=0; i<rgb_tstamps_tmp.size();i++){
			double min_d = fabs(rgb_tstamps_tmp[i]-d_tstamps_tmp[1]);
			int d_match_id = 0;
			for(int j=0; j<d_tstamps_tmp.size();j++){
				if (d_tstamps_tmp[j]-rgb_tstamps_tmp[i] > 0.06)
					break;
				double d = fabs(rgb_tstamps_tmp[i]-d_tstamps_tmp[j]);
				if (d<min_d){
					min_d = d;
					d_match_id = j;
				}
			}
			if(min_d<0.03){
				rgb_files.push_back(rgb_files_tmp[i]);
				rgb_tstamps.push_back(rgb_tstamps_tmp[i]);
				d_files.push_back(d_files_tmp[d_match_id]);
				d_tstamps.push_back(d_tstamps_tmp[d_match_id]);
			}
		}
	}
}

void Dataset::load_and_align_pose_gt(string filepath){

	FILE * pFile;
	stringstream pose_list_file;
	pose_list_file<<filepath<<"/groundtruth.txt";
	vector<double> pose_tstamps;
	vector<double*> poses;

	const string tmp = pose_list_file.str();
	const char* cstr = tmp.c_str();
	pFile = fopen(cstr,"r");
	if(pFile){
		char buff[500];

		//Skip first 3 lines
		fgets(buff,500,pFile);
		fgets(buff,500,pFile);
		fgets(buff,500,pFile);

		while(fgets(buff,500,pFile)!=NULL){

			char * head  = strtok(buff," ");
			double tstamp = strtod(head, NULL);
			pose_tstamps.push_back(tstamp);

			double *pose = new double[7];
			for(int i=0; i<7;i++){
				head  = strtok(NULL," \n");
				pose[i] = strtod(head, NULL);
			}
			poses.push_back(pose);
		}
		fclose(pFile);
	}
	else{
		cerr<<"Ground-truth poses missing"<<endl;
		return;
	}

	if (dataset_type==Synthetic){
		for(int i=0; i<poses.size(); i++){
			gt_poses.push_back(poses[i]);
			has_gt_pose.push_back(true);
		}
	}else{

		// Align the timestamps
		int last_assigned_id = -1; 
		for (int i=0; i<rgb_tstamps.size(); i++){
			double tstamp = rgb_tstamps[i];
			double min_offset = 10000000000000000000.0f;
			int best_match_id = -1;
			for(int j=last_assigned_id+1; j<pose_tstamps.size(); j++){
				double offset = pose_tstamps[j]-tstamp;
				if (offset<0.02){
					if (fabs(offset)<min_offset){
						best_match_id = j;
						min_offset = fabs(offset);
					}
				}
			}

			if(best_match_id>-1){
				gt_poses.push_back(poses[best_match_id]);
				has_gt_pose.push_back(true);
				last_assigned_id = best_match_id;
			}else{
				gt_poses.push_back(new double[7]);
				has_gt_pose.push_back(false);
			}
		}
	}
}


#include <fstream>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "viewer.h"
#include "visual_odometry.h"
#include "system.h"

myslam::Viewer *mpViewer;
std::thread *mptViewer;

void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames, std::vector<double> &vTimestamps)
{
	std::ifstream f;
	f.open(strFile.c_str());

	// skip first three lines
	std::string s0;
	getline(f, s0);
	getline(f, s0);
	getline(f, s0);

	int i = 0;

	while (!f.eof())
	{
		std::string s;
		getline(f, s);
		if (!s.empty())
		{
			std::stringstream ss;
			ss << s;
			double t;
			std::string sRGB;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenames.push_back(sRGB);
		}
	}
}

int main()
{
	std::string Imagepath = "/home/ubuntu/RGBD_SLAM_code/orb_slam_test/data/rgbd_dataset_freiburg1_xyz";
	std::string strImFile = "/home/ubuntu/RGBD_SLAM_code/orb_slam_test/data/rgbd_dataset_freiburg1_xyz/rgb.txt";
	std::string strDepthFile = "/home/ubuntu/RGBD_SLAM_code/orb_slam_test/data/rgbd_dataset_freiburg1_xyz/depth.txt";
	std::string strSettingPath = "/home/ubuntu/RGBD_SLAM_code/orb_slam_test/data/rgbd_dataset_freiburg1_xyz/cfg/TUM2.yaml";

	std::vector<std::string> vstrImageFilenames, vstrDImageFilenames;
	std::vector<double> vTimestamps, vDTimestamps;

	LoadImages(strImFile, vstrImageFilenames, vTimestamps);
	LoadImages(strDepthFile, vstrDImageFilenames, vDTimestamps);

	myslam::System System(strSettingPath);

	int nImages = vstrImageFilenames.size();

	cv::Mat im, imD;
	for (int ni = 0; ni < nImages; ni++)
	{
		im = cv::imread(std::string(Imagepath) + "/" + vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
		imD = cv::imread(std::string(Imagepath) + "/" + vstrDImageFilenames[ni], cv::IMREAD_UNCHANGED);

		double tframe = vTimestamps[ni];
		if (im.data == nullptr || imD.data == nullptr)
			break;

		cv::Mat T_c_w = System.TrackingRGBD(im, imD, tframe);

		cv::waitKey(10);
	}
}
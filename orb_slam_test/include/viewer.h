#ifndef VIEWER_H
#define VIEWER_H

#include <thread>
#include <mutex>
#include <iomanip>
#include<opencv2/core/core.hpp>

#include "visual_odometry.h"

#include <opencv2/opencv.hpp>
#include <pangolin//pangolin.h>

namespace myslam
{
	using namespace std;
	class VisualOdometry;
	class Viewer
	{
	public:
		Viewer(std::string strSettingPath);

		void Run();

		//获得当前帧位姿
		void SetCurrentCameraPose(const cv::Mat &Tcw);

		//获得当前帧跟踪信息
		void SetVisualOdometry(myslam::VisualOdometry* pVisualOdometry);

		//获得所有的关键帧数据
		void GetAllFrame(std::vector<Frame*> vAllFrame);

		//获得所有的三维点数据
		void GetAll3dPoints(std::vector<cv::Point3f> All3dpts);

		std::mutex mMutexViewer;

	private:
		void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
		void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

		void DrawTrackingFrame();
		void DrawMapPoints();

		cv::Mat DrawFrame();

		pangolin::View d_cam;
		pangolin::OpenGlMatrix Tcw;
		pangolin::OpenGlRenderState s_cam;

		std::mutex mMutexCamera;
		std::mutex mMutexTrackingFrame;
		std::mutex mMutex3dPoints;
		std::mutex mMutex;

		std::vector<cv::Point3f> mAll3dpts;

		std::vector<cv::KeyPoint> mkps;
		std::vector<cv::DMatch> mfeature_matches;

		std::vector<Frame*> mvAllFrame;

		cv::Mat mIm;
		int n;

		int state;

		// 1/fps in ms
		double mT;
		float mImageWidth, mImageHeight;

		float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

		float mKeyFrameSize;
		float mKeyFrameLineWidth;
		float mGraphLineWidth;
		float mPointSize;
		float mCameraSize;
		float mCameraLineWidth;

		cv::Mat mCameraPose;

		myslam::VisualOdometry* VisualOdometry_;
	};
}
#endif
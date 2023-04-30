#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include "common_include.h"
#include "map.h"
#include "viewer.h"
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace myslam
{
	class Viewer;
	class VisualOdometry
	{
	public:
		// 里程计状态
		enum VOState
		{
			INITIALZING = -1,
			OK = 0,
			LOST
		};

		VOState state_;

		// 地图
		Map *map_;

		// 图像帧,参考帧,当前帧
		Frame *ref_;
		Frame *curr_;

		// 所有的图像帧
		std::vector<Frame *> mvAllFrame;

		Viewer *mpViewer;
		Camera *mp_camera;

		cv::Ptr<cv::ORB> orb_;

		// 参考帧三维点,所有帧三维点
		std::vector<cv::Point3f> pts_3d_ref;
		std::vector<cv::Point3f> pts_3d_all;

		// 当前帧特征点
		std::vector<cv::KeyPoint> keypoints_curr_;

		// 当前帧、参考帧描述子
		cv::Mat descriptors_curr_, descriptors_ref_;

		// 特征匹配
		std::vector<cv::DMatch> feature_matches_;

		// 参考帧到当前帧的位姿信息
		cv::Mat T_c_r_estimate;

		// id
		int frameid;

		// 内点数量
		int num_inliers_;
		int num_lost_;

		// orb特征提取参数
		int num_of_features_;
		double scale_factor_;
		int level_pyramid_;
		float match_ratio_;
		int max_num_lost_;
		int min_inliers_;

		// 关键帧的最小旋转和平移
		double key_frame_min_rot;
		double key_frame_min_trans;

	public:
		VisualOdometry(std::string strSettingPath, Viewer *pViewer);
		~VisualOdometry();

		cv::Mat Tracking(cv::Mat im, cv::Mat imD, double tframe);

	private:
		cv::Mat addFrame(Frame *frame);

		void ExtractORB();
		void featureMatching();
		void poseEstimationPnP();
		void setRef3DPoints();

		void addKeyFrame();
		bool checkEstimatedPose();
		bool checkKeyFrame();
	};
}
#endif

#ifndef VISUAL_ODOMETRY_H
#define VISUAL_ODOMETRY_H

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "common_include.h"
#include "map.h"
#include "viewer.h"
#include "optimizer.h"
#include "orbfeatures.h"

namespace myslam
{
	using namespace ORB_SLAM2;

	class Viewer;
	class VisualOdometry
	{
	public:
		// ��̼�״̬
		enum VOState
		{
			INITIALZING = -1,
			OK = 0,
			LOST
		};

		VOState state_;

		// ��ͼ
		Map *map_;
		MapPoints *mappoints_;

		// ͼ��֡,�ο�֡,��ǰ֡
		Frame *ref_;
		Frame *curr_;

		// ���е�ͼ��֡
		std::vector<Frame *> mvAllFrame;

		Viewer *mpViewer;
		Camera *mp_camera;
		Converter *convert_;

		// ORB������ȡ
		ORBextractor *mp_ORBExtractor;

		// ����֡��ά��
		std::vector<cv::Point3f> pts_3d_all;

		// ��ǰ֡������
		std::vector<cv::KeyPoint> keypoints_curr_;

		// ��ǰ֡���ο�֡������
		cv::Mat descriptors_curr_, descriptors_ref_;

		// ����ƥ��
		std::vector<cv::DMatch> feature_matches_;

		// ��������ϵ���������ϵ��λ����Ϣ
		cv::Mat T_c_w_estimate;

		cv::Mat mK;

		// id
		int frameid;

		// �ڵ�����
		int num_inliers_;
		int num_lost_;

		// orb������ȡ����
		int num_of_features_;
		double scale_factor_;
		int level_pyramid_;
		int iniThFAST;
		int minThFAST;
		float match_ratio_;
		int max_num_lost_;
		int min_inliers_;

		double map_point_erase_ratio_;

		// �ؼ�֡����С��ת��ƽ��
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

		void addKeyFrame();
		bool checkEstimatedPose();
		bool checkKeyFrame();
		void optimizeMap();
		void addMapPoints();

		// ��õ�ͼ��P�ڵ�ǰ֡����ϵO1����������ԭ��O2֮���γɵļн�
		double getViewAngle(Frame *frame, MapPoints *point);
	};
}
#endif
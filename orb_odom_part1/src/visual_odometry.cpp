#include "visual_odometry.h"

namespace myslam
{
	VisualOdometry::VisualOdometry(std::string strSettingPath, Viewer *pViewer) : state_(INITIALZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0), mpViewer(pViewer)
	{
		// 读取配置文件
		cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

		num_of_features_ = fSettings["number_of_features"];
		scale_factor_ = fSettings["scale_factor"];
		level_pyramid_ = fSettings["level_pyramid"];
		match_ratio_ = fSettings["match_ratio"];
		max_num_lost_ = fSettings["max_num_lost"];
		min_inliers_ = fSettings["min_inliers"];
		key_frame_min_rot = fSettings["keyframe_rotation"];
		key_frame_min_trans = fSettings["keyframe_translation"];
		fSettings.release();

		// 构建相机类
		mp_camera = new myslam::Camera(strSettingPath);

		// 构建orb特征提取类
		orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);

		frameid = 0;
	}

	VisualOdometry::~VisualOdometry()
	{
	}

	// 跟踪
	cv::Mat VisualOdometry::Tracking(cv::Mat im, cv::Mat imD, double tframe)
	{
		// 构建图像帧
		Frame *pFrame = new Frame(frameid);
		pFrame->color_ = im;
		pFrame->depth_ = imD;
		pFrame->camera_ = mp_camera;
		pFrame->time_stamp_ = tframe;

		// 获得跟踪位置信息
		cv::Mat T_c_w = addFrame(pFrame);

		frameid++;

		return T_c_w;
	}

	cv::Mat VisualOdometry::addFrame(Frame *frame)
	{
		cv::Mat T_c_w;

		// 未初始化
		if (state_ == INITIALZING)
		{
			std::unique_lock<std::mutex> lock(mpViewer->mMutexViewer);

			// 当前帧,参考帧构建
			curr_ = ref_ = frame;
			ref_->T_c_w_ = cv::Mat::eye(4, 4, CV_32FC1);
			curr_->T_c_w_ = cv::Mat::eye(4, 4, CV_32FC1);

			// 当前帧插入地图
			map_->insertKeyFrame(frame);

			// 提取ORB特征
			ExtractORB();

			// 获得特征点对应的三维点
			setRef3DPoints();

			state_ = OK;
			return curr_->T_c_w_;
		}
		else
		{
			// 跟踪丢失
			if (state_ == LOST)
			{
				std::cout << "vo has lost." << std::endl;

				num_lost_++;
				if (num_lost_ > max_num_lost_)
				{
					state_ = LOST;
				}
				return ref_->T_c_w_;
			}
			else
			{
				// 位姿跟踪
				curr_ = frame;
				ExtractORB();

				// 特征匹配
				featureMatching();

				// 估计上一帧与当前帧的位置关系
				poseEstimationPnP();

				// 检测位姿是否正确
				if (checkEstimatedPose() == true)
				{
					// 计算当前帧相对于世界坐标系的位姿
					curr_->T_c_w_ = T_c_r_estimate * ref_->T_c_w_;
					ref_ = curr_;

					// 计算当前帧的三维点
					setRef3DPoints();
					num_lost_ = 0;

					// 是否是关键帧
					if (checkKeyFrame() == true)
					{
						addKeyFrame();
					}
				}

				// 所有的图像帧
				mvAllFrame.push_back(curr_);

				cv::Mat T_c_w = curr_->T_c_w_;

				// 向显示类中添加相信的数据
				mpViewer->SetCurrentCameraPose(T_c_w);
				mpViewer->GetAllFrame(mvAllFrame);
				mpViewer->GetAll3dPoints(pts_3d_all);
				mpViewer->SetVisualOdometry(this);

				cv::waitKey(10);

				return T_c_w;
			}
		}
	}

	// 提取特征点与描述子
	void VisualOdometry::ExtractORB()
	{
		orb_->detectAndCompute(curr_->color_, cv::Mat(), keypoints_curr_, descriptors_curr_);
	}

	// 特征匹配，BF+距离阈值筛选
	void VisualOdometry::featureMatching()
	{
		std::vector<cv::DMatch> matches;
		cv::BFMatcher matcher(cv::NORM_HAMMING);

		matcher.match(descriptors_ref_, descriptors_curr_, matches);
		float max_dis = 0;
		for (int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance > max_dis)
				max_dis = matches[i].distance;
		}

		feature_matches_.clear();
		for (cv::DMatch &m : matches)
		{
			if (m.distance < max_dis * 0.3)
				feature_matches_.push_back(m);
		}
		std::cout << "good matches: " << feature_matches_.size() << std::endl;
	}

	// 计算每个特征点对应的三维点
	void VisualOdometry::setRef3DPoints()
	{
		pts_3d_ref.clear();
		descriptors_ref_ = cv::Mat();
		for (int i = 0; i < keypoints_curr_.size(); i++)
		{
			double d = ref_->findDepth(keypoints_curr_[i]);

			cv::Point3f p_cam = ref_->camera_->pixel2camera(cv::Point2f(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d);
			pts_3d_ref.push_back(p_cam);
			pts_3d_all.push_back(p_cam);
			descriptors_ref_.push_back(descriptors_curr_.row(i));
		}
	}

	void VisualOdometry::poseEstimationPnP()
	{
		std::vector<cv::Point3f> pts3d;
		std::vector<cv::Point2f> pts2d;

		// 上一帧与当前帧的匹配点
		for (int i = 0; i < feature_matches_.size(); i++)
		{
			if (pts_3d_ref[feature_matches_[i].queryIdx].z < 0)
				continue;

			// 上一帧的三维点
			pts3d.push_back(pts_3d_ref[feature_matches_[i].queryIdx]);

			// 该三维点在当前帧对应的二维点
			pts2d.push_back(keypoints_curr_[feature_matches_[i].trainIdx].pt);
		}

		// 内参
		cv::Mat K = (cv::Mat_<float>(3, 3) << ref_->camera_->fx_, 0.0, ref_->camera_->cx_,
					 0.0, ref_->camera_->fy_, ref_->camera_->cy_,
					 0.0, 0.0, 1.0);

		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32FC1);
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32FC1);
		std::vector<int> inliers;

		// pnp位姿计算
		cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
		num_inliers_ = inliers.size();
		std::cout << "pnp inliers: " << inliers.size() << std::endl;

		cv::Rodrigues(rvec, rvec);

		// 获得参考帧相对于当前帧的位置关系
		T_c_r_estimate = cv::Mat::eye(4, 4, CV_32FC1);
		rvec.copyTo(T_c_r_estimate(cv::Rect(0, 0, 3, 3)));
		tvec.copyTo(T_c_r_estimate(cv::Rect(3, 0, 1, 3)));
	}

	void VisualOdometry::addKeyFrame()
	{
		std::cout << "adding a key-frame" << std::endl;
		map_->insertKeyFrame(curr_);
	}

	bool VisualOdometry::checkEstimatedPose()
	{
		// 内点数太少
		if (num_inliers_ < min_inliers_)
		{
			std::cout << "reject because inlier is too small: " << num_inliers_ << std::endl;
			return false;
		}

		cv::Mat rvec = T_c_r_estimate(cv::Rect(0, 0, 3, 3));
		cv::Mat tvec = T_c_r_estimate(cv::Rect(3, 0, 1, 3));

		// 移动距离太大或是旋转矩阵计算错误
		if (tvec.at<double>(0, 0) > 20.0 || abs(1.0 - determinant(rvec)) > 0.01)
		{
			std::cout << "reject because motion is too large: " << std::endl;
			return false;
		}
		return true;
	}

	bool VisualOdometry::checkKeyFrame()
	{
		cv::Mat rvec = T_c_r_estimate(cv::Rect(0, 0, 3, 3));
		cv::Mat tvec = T_c_r_estimate(cv::Rect(3, 0, 1, 3));

		// 计算旋转角度
		cv::Scalar t = cv::trace(rvec);
		double trR = t.val[0];
		double theta = acos((trR - 1.0) / 2.0);

		// 超过最小的旋转或是平移阈值
		if (abs(theta) > key_frame_min_rot || norm(tvec) > key_frame_min_trans)
			return true;
		return false;
	}
}

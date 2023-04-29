#include "visual_odometry.h"

namespace myslam
{
	VisualOdometry::VisualOdometry(std::string strSettingPath, Viewer* pViewer) :
		state_(INITIALZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0), mpViewer(pViewer)
	{
		//��ȡ�����ļ�
		cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

		num_of_features_ = fSettings["number_of_features"];
		scale_factor_ = fSettings["scale_factor"];
		level_pyramid_ = fSettings["level_pyramid"];
		iniThFAST = fSettings["ORBextractor.iniThFAST"];
		minThFAST = fSettings["ORBextractor.minThFAST"];

		match_ratio_ = fSettings["match_ratio"];
		max_num_lost_ = fSettings["max_num_lost"];
		min_inliers_ = fSettings["min_inliers"];
		key_frame_min_rot = fSettings["keyframe_rotation"];
		key_frame_min_trans = fSettings["keyframe_translation"];
		map_point_erase_ratio_ = fSettings["map_point_erase_ratio"];

		fSettings.release();

		//���������
		mp_camera = new myslam::Camera(strSettingPath);

		//�ڲ�
		mK = (cv::Mat_<float>(3, 3) << mp_camera->fx_, 0.0, mp_camera->cx_,
			0.0, mp_camera->fy_, mp_camera->cy_,
			0.0, 0.0, 1.0);

		//����orb������ȡ
		mp_ORBExtractor = new ORBextractor(num_of_features_, scale_factor_, level_pyramid_, iniThFAST, minThFAST);

		frameid = 0;
	}

	VisualOdometry::~VisualOdometry()
	{

	}

	//����
	cv::Mat VisualOdometry::Tracking(cv::Mat im, cv::Mat imD, double tframe)
	{
		//����ͼ��֡
		Frame* pFrame = new Frame(frameid);
		pFrame->color_ = im;
		pFrame->depth_ = imD;
		pFrame->camera_ = mp_camera;
		pFrame->time_stamp_ = tframe;

		//��ø���λ����Ϣ
		cv::Mat T_c_w = addFrame(pFrame);

		frameid++;

		return T_c_w;
	}

	cv::Mat VisualOdometry::addFrame(Frame* frame)
	{
		//δ��ʼ��
		if (state_ == INITIALZING)
		{
			std::unique_lock<std::mutex> lock(mpViewer->mMutexViewer);

			//��ǰ֡,�ο�֡����
			curr_ = ref_ = frame;
			ref_->T_c_w_ = cv::Mat::eye(4, 4, CV_32FC1);
			curr_->T_c_w_ = cv::Mat::eye(4, 4, CV_32FC1);

			//��ȡORB����
			ExtractORB();

			addKeyFrame();

			state_ = OK;
			return curr_->T_c_w_;
		}
		else
		{
			//���ٶ�ʧ
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
				//λ�˸���
				curr_ = frame;
				curr_->T_c_w_ = ref_->T_c_w_;

				ExtractORB();

				//����ƥ��
				featureMatching();

				//������һ֡�뵱ǰ֡��λ�ù�ϵ
				poseEstimationPnP();

				//���λ���Ƿ���ȷ
				if (checkEstimatedPose() == true)
				{
					//���㵱ǰ֡�������������ϵ��λ��
					curr_->T_c_w_ = T_c_w_estimate;

					ref_ = curr_;
					num_lost_ = 0;

					optimizeMap();

					//�Ƿ��ǹؼ�֡
					if (checkKeyFrame() == true)
					{
						addKeyFrame();
					}
				}

				//���е�ͼ��֡
				mvAllFrame.push_back(curr_);

				cv::Mat T_c_w = curr_->T_c_w_;

				//����ʾ����������ŵ�����
				mpViewer->SetCurrentCameraPose(T_c_w);
				mpViewer->GetAllFrame(mvAllFrame);
				mpViewer->GetAll3dPoints(pts_3d_all);
				mpViewer->SetVisualOdometry(this);

				cv::waitKey(10);

				return T_c_w;
			}
		}
	}

	//��ȡ��������������
	void VisualOdometry::ExtractORB()
	{
		mp_ORBExtractor->runORBextractor(curr_->color_, keypoints_curr_, descriptors_curr_);
	}

	//����ƥ�䣬BF+������ֵɸѡ
	void VisualOdometry::featureMatching()
	{
		std::vector<cv::DMatch> matches;
		cv::BFMatcher matcher(cv::NORM_HAMMING);

		cv::Mat descriptors_map;
		std::vector<MapPoints*> candidate;
		for (auto& it: map_->map_points_ )
		{
			MapPoints* mappoints = it.second;
			if (curr_->isInFrame(mappoints->pos_))
			{
				it.second->visible_times_++;
				candidate.push_back(mappoints);
				descriptors_map.push_back(mappoints->descriptor_);
			}
		}

		matcher.match(descriptors_map, descriptors_curr_, matches);
		float max_dis = 0;
		for (int i = 0; i < matches.size(); i++)
		{
			if (matches[i].distance > max_dis)
				max_dis = matches[i].distance;
		}

		//candidate�е�ƥ������ı��ˣ�map_points_�е�ƥ�����Ҳ�ı���
		feature_matches_.clear();
		for (cv::DMatch& m : matches)
		{
			if (m.distance < max_dis*0.2)
			{
				curr_->m_vpts3d.push_back(candidate[m.queryIdx]->pos_);
				curr_->m_vpts2d.push_back(keypoints_curr_[m.trainIdx].pt);

				feature_matches_.push_back(m);
				candidate[m.queryIdx]->matched_times_++;
			}
		}
		std::cout << "good matches: " << curr_->m_vpts3d.size() << std::endl;
	}

	void VisualOdometry::poseEstimationPnP()
	{
		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_32FC1);
		cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32FC1);
		std::vector<int> inliers;

		//pnpλ�˼���
		cv::solvePnPRansac(curr_->m_vpts3d, curr_->m_vpts2d, mK, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
		num_inliers_ = inliers.size();
		std::cout << "pnp inliers: " << inliers.size() << std::endl;

		cv::Rodrigues(rvec, rvec);

		//��òο�֡����ڵ�ǰ֡��λ�ù�ϵ
		T_c_w_estimate = cv::Mat::eye(4, 4, CV_32FC1);
		rvec.copyTo(T_c_w_estimate(cv::Rect(0, 0, 3, 3)));
		tvec.copyTo(T_c_w_estimate(cv::Rect(3, 0, 1, 3)));

		Optimizer optimizer;
		optimizer.PoseOptimization(curr_, inliers, T_c_w_estimate);
	}

	void VisualOdometry::addKeyFrame()
	{
		std::cout << "adding a key-frame" << std::endl;

		cv::Point3d p_world, n;
		for (int i = 0; i < keypoints_curr_.size(); i++)
		{
			double d = curr_->findDepth(keypoints_curr_[i]);
			if (d < 0 || d > 5.0)
				continue;

			p_world = curr_->camera_->pixel2world(cv::Point2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), curr_->T_c_w_.inv(), d);
			n = p_world - cv::Point3d(curr_->getCamCenter());

			pts_3d_all.push_back(p_world);
			
			n = n / norm(n);
			MapPoints* map_point = mappoints_->createMapPoints(p_world, n, descriptors_curr_.row(i), curr_);

			map_->insertMapPoint(map_point);
		}
		map_->insertKeyFrame(curr_);
		ref_ = curr_;
	}

	bool VisualOdometry::checkEstimatedPose()
	{
		//�ڵ���̫��
		if (num_inliers_ < min_inliers_)
		{
			std::cout << "reject because inlier is too small: " << num_inliers_ << std::endl;
			return false;
		}

		cv::Mat rvec = T_c_w_estimate(cv::Rect(0, 0, 3, 3));
		cv::Mat tvec = T_c_w_estimate(cv::Rect(3, 0, 1, 3));

		//�ƶ�����̫�������ת����������
		if (tvec.at<double>(0, 0) > 20.0 || abs(1.0 - determinant(rvec)) > 0.01)
		{
			std::cout << "reject because motion is too large: " << std::endl;
			return false;
		}
		return true;
	}

	bool VisualOdometry::checkKeyFrame()
	{
		cv::Mat rvec = T_c_w_estimate(cv::Rect(0, 0, 3, 3));
		cv::Mat tvec = T_c_w_estimate(cv::Rect(3, 0, 1, 3));

		//������ת�Ƕ�
		cv::Scalar t = cv::trace(rvec);
		double trR = t.val[0];
		double theta = acos((trR - 1.0) / 2.0);

		//������С����ת����ƽ����ֵ
		if (abs(theta) > key_frame_min_rot || norm(tvec) > key_frame_min_trans)
			return true;
		return false;
	}

	void VisualOdometry::optimizeMap()
	{
		for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); iter++)
		{
			if (!curr_->isInFrame(iter->second->pos_))
			{
				iter = map_->map_points_.erase(iter);
				continue;
			}
			float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
			if (match_ratio < map_point_erase_ratio_)
			{
				iter = map_->map_points_.erase(iter);
				continue;
			}

			double angle = getViewAngle(curr_, iter->second);

			//�Ƕȴ���10�ȣ��ھֲ���ͼ��ɾ���õ�ͼ��
			if (angle > M_PI / 36.)
			{
				iter = map_->map_points_.erase(iter);
				continue;
			}
			if (iter->second->good_ == false)
			{
				// TODO try triangulate this map point 
			}
		}

		if (feature_matches_.size() < 200)
			addMapPoints();

		if (map_->map_points_.size() > 1000)
			map_point_erase_ratio_ += 0.05;
		else
			map_point_erase_ratio_ = 0.1;

		cout << "map points: " << map_->map_points_.size() << endl;
	}

	void VisualOdometry::addMapPoints()
	{
		std::vector<bool> matched(keypoints_curr_.size(), false);
		for (auto m: feature_matches_)
		{
			matched[m.trainIdx] = true;
		}

		for (int i = 0; i < keypoints_curr_.size(); i++)
		{
			if (matched[i] ==  true)
				continue;

			double d = curr_->findDepth(keypoints_curr_[i]);
			if (d < 0 || d > 5.0)
				continue;

			cv::Point3d p_world = curr_->camera_->pixel2world(keypoints_curr_[i].pt, curr_->T_c_w_.inv(), d);
			cv::Point3d n = p_world - cv::Point3d(curr_->getCamCenter());

			n = n / norm(n);
			MapPoints* mappoint = mappoints_->createMapPoints(p_world, n, descriptors_curr_.row(i).clone(), curr_);
			map_->insertMapPoint(mappoint);
		}
	}

	double VisualOdometry::getViewAngle(Frame* frame, MapPoints* point)
	{
		cv::Mat temp = frame->getCamCenter();
		cv::Point3d n = point->pos_ - cv::Point3d(frame->getCamCenter());

// 		float a = n.x*point->pos_.x + n.y*point->pos_.y + n.z*point->pos_.z;
// 		float b = norm(n);
// 		float c = norm(point->pos_);

		//return acos(a/(b*c));

		n = n / norm(n);
		return acos(n.x*point->norm_.x + n.y*point->norm_.y + n.z*point->norm_.z);

	}
}
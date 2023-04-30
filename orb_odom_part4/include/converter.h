#ifndef CONVERTER_H
#define CONVERTER_H

#include "../Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "../Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace myslam
{
	class Converter
	{
	public:
		std::vector<cv::Mat> toDescriptorVector(cv::Mat Descriptors);

		g2o::SE3Quat toSE3Quat(cv::Mat cvT);
		g2o::SE3Quat toSE3Quat(g2o::Sim3 gSim3);

		cv::Mat toCvMat(g2o::SE3Quat SE3);
		cv::Mat toCvMat(g2o::Sim3 Sim3);
		cv::Mat toCvMat(Eigen::Matrix<double, 4, 4> m);
		cv::Mat toCvMat(Eigen::Matrix3d m);
		cv::Mat toCvMat(Eigen::Matrix<double, 3, 1> m);
		cv::Mat toCvSE3(Eigen::Matrix<double, 3, 3> R, Eigen::Matrix<double, 3, 1> t);

		Eigen::Matrix<double, 3, 1> toVector3d(cv::Mat cvVector);
		Eigen::Matrix<double, 3, 1> toVector3d(cv::Point3f cvPoint);
		Eigen::Matrix<double, 2, 1> toVector2d(cv::Point2f cvPoint);
		Eigen::Matrix<double, 3, 3> toMatrix3d(cv::Mat cvMat3);

		cv::Point3f toPoint3f(Eigen::Vector3d v);

		std::vector<float> toQuaternion(cv::Mat M);
	};

}
#endif // CONVERTER_H

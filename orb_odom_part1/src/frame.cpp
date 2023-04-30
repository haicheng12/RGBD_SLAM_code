#include "frame.h"

namespace myslam
{
	Frame::Frame() : id_(-1), time_stamp_(-1), camera_(nullptr) {}
	Frame::Frame(long id, double time_stamp, cv::Mat T_c_w, Camera *camera, cv::Mat color, cv::Mat depth) : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
	{
	}

	double Frame::findDepth(const cv::KeyPoint &kp)
	{
		int x = round(kp.pt.x);
		int y = round(kp.pt.y);

		ushort d = depth_.ptr<ushort>(y)[x];
		if (d != 0)
			return d / camera_->depth_scale_;
		else
		{
			// 没有找到合适的深度值时,在其四领域内再次查找
			int dx[4] = {-1, 0, 1, 0};
			int dy[4] = {0, -1, 0, 1};
			for (int i = 0; i < 4; i++)
			{
				d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
				if (d != 0)
					return d / camera_->depth_scale_;
			}
		}

		return -1.0;
	}

	cv::Mat Frame::getCamCenter() const
	{
		cv::Mat T_w_c = T_c_w_.inv(cv::DecompTypes::DECOMP_SVD);

		return T_w_c(cv::Rect(3, 0, 1, 3));
	}

	bool Frame::isInFrame(const cv::Point3d pt_world)
	{
		cv::Point3d p_cam = camera_->world2camera(pt_world, T_c_w_);
		if (p_cam.z < 0)
			return false;

		cv::Point2d pixel = camera_->camera2pixel(p_cam);

		return (pixel.x > 0 && pixel.y > 0 && pixel.x < color_.cols && pixel.y < color_.rows);
	}
}

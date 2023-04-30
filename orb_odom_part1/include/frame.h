#ifndef FRAME_H
#define FRAME_H

#include "common_include.h"
#include "camera.h"
#include "mappoints.h"

namespace myslam
{
	class MapPoints;
	class Frame
	{
	public:
		unsigned long id_;		// 图像帧的id
		double time_stamp_;		// 时间戳
		cv::Mat T_c_w_;			// 世界坐标系到相机坐标系的变换关系
		Camera *camera_;		// 相机类指针
		cv::Mat color_, depth_; // 彩图与深度图
	public:
		// 默认构造函数
		Frame();

		// 构造图像帧
		Frame(long id, double time_stamp = 0, cv::Mat T_c_w = cv::Mat(), Camera *camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());

		// 找到某个特征点在深度图中对应的深度值
		double findDepth(const cv::KeyPoint &kp);

		// 获得相机坐标系到世界坐标系的平移向量
		cv::Mat getCamCenter() const;

		// 判断某个三维点是否在图像的视野范围内
		bool isInFrame(const cv::Point3d pt_world);
	};
}

#endif

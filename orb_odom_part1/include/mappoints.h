#ifndef MAPPOINTS_H
#define MAPPOINTS_H

#include "frame.h"

namespace myslam
{
	class Frame;
	class MapPoints
	{
	public:
		MapPoints();
		MapPoints(long id, cv::Point3d position, cv::Point3d norm);

	public:
		unsigned long id_;

		cv::Point3d pos_;  // 位置
		cv::Point3d norm_; // 方向

		cv::Mat descriptor;	 // 描述子
		int observed_times_; // 观测次数
		int correct_times_;	 // 校正次数

	public:
		MapPoints *creatMappoints();
	};
}
#endif

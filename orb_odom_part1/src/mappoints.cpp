#include "mappoints.h"

namespace myslam
{
	MapPoints::MapPoints() : id_(-1), pos_(cv::Point3d(0.0, 0.0, 0.0)), observed_times_(0), correct_times_(0) {}
	MapPoints::MapPoints(long id, cv::Point3d position, cv::Point3d norm) : id_(id), pos_(position), norm_(norm), observed_times_(0), correct_times_(0) {}

	MapPoints *MapPoints::creatMappoints()
	{
		static long factory_id = 0;
		return (new MapPoints(factory_id, cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(0.0, 0.0, 0.0)));
	}
}

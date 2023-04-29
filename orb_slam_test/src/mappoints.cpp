#include "mappoints.h"

static unsigned long factory_id_ = 0;    // factory id

namespace myslam
{
	MapPoints::MapPoints() :id_(-1), pos_(cv::Point3d(0.0, 0.0, 0.0)), matched_times_(0), visible_times_(0) {}
	MapPoints::MapPoints(unsigned long id, cv::Point3d position, cv::Point3d  norm, Frame* frame, cv::Mat descriptor)
		:id_(id), pos_(position), norm_(norm), good_(true), visible_times_(1), matched_times_(1), descriptor_(descriptor.clone())
	{
		observed_frames_.push_back(frame);
	}

	MapPoints* MapPoints::createMapPoints()
	{
		return (new MapPoints(factory_id_++, cv::Point3d(0.0, 0.0, 0.0), cv::Point3d(0.0, 0.0, 0.0) ));
	}

	MapPoints* MapPoints::createMapPoints(cv::Point3d pos_world, cv::Point3d norm, cv::Mat descriptor, Frame* frame)
	{
		return(new MapPoints(factory_id_++, pos_world, norm, frame, descriptor));
	}
} 
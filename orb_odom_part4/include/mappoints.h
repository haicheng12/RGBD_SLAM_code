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
		MapPoints(unsigned long id, cv::Point3d position, cv::Point3d norm, Frame *frame = nullptr, cv::Mat descriptor = cv::Mat());

	public:
		unsigned long id_;	 // ID
		bool good_;			 // wheter a good point
		cv::Point3d pos_;	 // Position in world
		cv::Point3d norm_;	 // Normal of viewing direction
		cv::Mat descriptor_; // Descriptor for matching

		std::list<Frame *> observed_frames_; // key-frames that can observe this point

		int matched_times_; // being an inliner in pose estimation
		int visible_times_; // being visible in current frame

	public:
		inline cv::Point3f getPositionCV() const
		{
			return pos_;
		}

		MapPoints *createMapPoints();
		MapPoints *createMapPoints(cv::Point3d pos_world, cv::Point3d norm, cv::Mat descriptor, Frame *frame);
	};
}
#endif
#ifndef MAP_H
#define MAP_H

#include "common_include.h"
#include "frame.h"
#include "mappoints.h"

namespace myslam
{
	class Map
	{
	public:
		std::unordered_map<unsigned long, MapPoints *> map_points_; // 所有的地图点
		std::unordered_map<unsigned long, Frame *> keyframe_;		// 所有的关键帧

		Map() {}

		// 插入关键帧
		void insertKeyFrame(Frame *frame);

		// 插入地图点
		void insertMapPoint(MapPoints *map_points);
	};
}

#endif

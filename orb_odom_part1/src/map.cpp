#include "map.h"

namespace myslam
{
	void Map::insertKeyFrame(Frame *frame)
	{
		std::cout << "Key frame size = " << keyframe_.size() << std::endl;
		if (keyframe_.find(frame->id_) == keyframe_.end())
		{
			keyframe_.insert(std::make_pair(frame->id_, frame));
		}
		else
		{
			keyframe_[frame->id_] = frame;
		}
	}

	void Map::insertMapPoint(MapPoints *map_points)
	{
		if (map_points_.find(map_points->id_) == map_points_.end())
		{
			map_points_.insert(std::make_pair(map_points->id_, map_points));
		}
		else
		{
			map_points_[map_points->id_] = map_points;
		}
	}
}

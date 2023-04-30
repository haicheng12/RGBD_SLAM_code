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
		std::unordered_map<unsigned long, MapPoints *> map_points_; // ���еĵ�ͼ��
		std::unordered_map<unsigned long, Frame *> keyframe_;		// ���еĹؼ�֡

		Map() {}

		// ����ؼ�֡
		void insertKeyFrame(Frame *frame);

		// �����ͼ��
		void insertMapPoint(MapPoints *map_points);
	};
}

#endif
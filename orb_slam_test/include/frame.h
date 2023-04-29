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
		unsigned long id_;         //ͼ��֡��id
		double time_stamp_;        //ʱ���
		cv::Mat T_c_w_;            //��������ϵ���������ϵ�ı任��ϵ
		Camera* camera_;           //�����ָ��
		cv::Mat color_, depth_;    //��ͼ�����ͼ

		std::vector<cv::Point3f> m_vpts3d;  //��ǰ֡���ٵ�����ά��
		std::vector<cv::Point2f> m_vpts2d;  //��ǰ֡���ٵ��Ķ�ά��
	public:
		//Ĭ�Ϲ��캯��
		Frame();

		//����ͼ��֡
		Frame(long id, double time_stamp = 0, cv::Mat T_c_w = cv::Mat(), Camera* camera = nullptr, cv::Mat color = cv::Mat(), cv::Mat depth = cv::Mat());

		//�ҵ�ĳ�������������ͼ�ж�Ӧ�����ֵ
		double findDepth(cv::KeyPoint kp);

		//����������ϵ����������ϵ��ƽ������
		cv::Mat getCamCenter();

		//�ж�ĳ����ά���Ƿ���ͼ�����Ұ��Χ��
		bool isInFrame(cv::Point3d pt_world);
	};
}

#endif
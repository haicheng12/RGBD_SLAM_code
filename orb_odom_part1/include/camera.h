#ifndef CAMERA_H
#define CAMERA_H

#include "common_include.h"

namespace myslam
{
	class Camera
	{
	public:
		Camera(std::string strSettingPath);
		Camera(float fx, float fy, float cx, float cy, float depth_scale) : fx_(fx), fy_(fy), cx_(cx), cy_(cy), depth_scale_(depth_scale) {}

		/**
		 *@brief world2camera 将世界坐标系下的三维点转换到相机坐标系下
		 *@param 输入 p_w：世界坐标系下的三维点
		 *@param 输入 T_c_w：世界坐标系到相机坐标系的变换关系
		 *@return  转换后的三维点
		 */
		cv::Point3d world2camera(cv::Point3d p_w, cv::Mat T_c_w);

		/**
		 *@brief camera2world 将相机坐标系下的三维点转换到世界坐标系下
		 *@param 输入 p_c：相机坐标系下的三维点
		 *@param 输入 T_w_c：相机坐标系到世界坐标系的变换关系
		 *@return  转换后的三维点
		 */
		cv::Point3d camera2world(cv::Point3d p_c, cv::Mat T_w_c);

		/**
		 *@brief camera2pixel 将相机坐标系下的三维点转换到图像坐标系下
		 *@param 输入 p_c：相机坐标系下的三维点
		 *@return  转换后的图像坐标系下的二维点
		 */
		cv::Point2d camera2pixel(cv::Point3d p_c);

		/**
		 *@brief pixel2camera 将图像坐标系下的二维点转换到相机坐标系下
		 *@param 输入 p_p：图像坐标系下的二维点
		 *@param 输入 depth：该二维点在相机坐标系下的深度值
		 *@return  转换后的相机坐标系下的三维点
		 */
		cv::Point3d pixel2camera(cv::Point2d p_p, double depth = 1.0);

		/**
		 *@brief pixel2world 将图像坐标系下的二维点转换到世界坐标系下
		 *@param 输入 p_p：图像坐标系下的二维点
		 *@param 输入 T_w_c：相机坐标系到世界坐标系的变换关系
		 *@param 输入 depth：该二维点在相机坐标系下的深度值
		 *@return  转换后的世界坐标系下的三维点
		 */
		cv::Point3d pixel2world(cv::Point2d p_p, cv::Mat T_w_c, double depth = 1.0);

		/**
		 *@brief world2pixel 将世界坐标系下的三维点转换到图像坐标系下
		 *@param 输入 p_w：世界坐标系下的三维点
		 *@param 输入 T_c_w：世界坐标系到相机坐标系的变换关系
		 *@return  转换后的图像坐标系下的二维点
		 */
		cv::Point2d world2pixel(cv::Point3d p_w, cv::Mat T_c_w);

	public:
		// 相机内部参数
		float fx_, fy_, cx_, cy_;

		// 将深度图转换为实际深度信息的尺度值
		float depth_scale_;
	};
}
#endif

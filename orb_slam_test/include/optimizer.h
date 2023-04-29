#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "../Thirdparty/g2o/g2o/core/block_solver.h"
#include "../Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "../Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "../Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "../Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "../Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "../Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include "common_include.h"
#include "camera.h"
#include "converter.h"
#include "frame.h"

namespace myslam
{
	class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		virtual void computeError();
		virtual void linearizeOplus();
		virtual bool read(std::istream &in) { return true; }
		virtual bool write(std::ostream &out) const { return true; }
	};

	// only to optimize the pose, no point
	class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		// Error: measure = R*point+t

		virtual void computeError();
		virtual void linearizeOplus();

		virtual bool read(std::istream &in) { return true; }
		virtual bool write(std::ostream &out) const { return true; }

		Eigen::Vector3d point_;
	};

	class EdgeProjectXYZ2UVPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		virtual void computeError();
		virtual void linearizeOplus();

		virtual bool read(std::istream &in) { return true; }
		virtual bool write(std::ostream &os) const { return true; }

		Eigen::Vector3d point_;
		Camera *camera_;
		Converter *convert_;
	};

	class Optimizer
	{
	public:
		Optimizer(){};

		void LocalBundleAdjustment(Frame *curr_, std::vector<int> inliers, cv::Mat &T_c_r_estimate);

		/**
		 * @brief Pose Only Optimization
		 *
		 * 3D-2D ��С����ͶӰ��� e = (u,v) - project(Tcw*Pw) \n
		 * ֻ�Ż�Frame��Tcw�����Ż�MapPoints������
		 *
		 * 1. Vertex: g2o::VertexSE3Expmap()������ǰ֡��Tcw
		 * 2. Edge:
		 *     - g2o::EdgeSE3ProjectXYZOnlyPose()��BaseUnaryEdge
		 *         + Vertex�����Ż���ǰ֡��Tcw
		 *         + measurement��MapPoint�ڵ�ǰ֡�еĶ�άλ��(u,v)
		 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
		 *     - g2o::EdgeStereoSE3ProjectXYZOnlyPose()��BaseUnaryEdge
		 *         + Vertex�����Ż���ǰ֡��Tcw
		 *         + measurement��MapPoint�ڵ�ǰ֡�еĶ�άλ��(ul,v,ur)
		 *         + InfoMatrix: invSigma2(�����������ڵĳ߶��й�)
		 *
		 * @param   pFrame Frame
		 * @return  inliers����
		 */
		void PoseOptimization(Frame *curr_, std::vector<int> inliers, cv::Mat &T_c_r_estimate);

	private:
		Converter *convert_;
	};
}

#endif
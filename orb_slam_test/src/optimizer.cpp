#include "optimizer.h"

namespace myslam
{
	void EdgeProjectXYZRGBD::computeError()
	{
		const g2o::VertexSBAPointXYZ* point = static_cast<const g2o::VertexSBAPointXYZ*> (_vertices[0]);
		const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[1]);
		_error = _measurement - pose->estimate().map(point->estimate());
	}

	void EdgeProjectXYZRGBD::linearizeOplus()
	{
		g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *> (_vertices[1]);
		g2o::SE3Quat T(pose->estimate());
		g2o::VertexSBAPointXYZ* point = static_cast<g2o::VertexSBAPointXYZ*> (_vertices[0]);
		Eigen::Vector3d xyz = point->estimate();
		Eigen::Vector3d xyz_trans = T.map(xyz);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];

		_jacobianOplusXi = -T.rotation().toRotationMatrix();

		_jacobianOplusXj(0, 0) = 0;
		_jacobianOplusXj(0, 1) = -z;
		_jacobianOplusXj(0, 2) = y;
		_jacobianOplusXj(0, 3) = -1;
		_jacobianOplusXj(0, 4) = 0;
		_jacobianOplusXj(0, 5) = 0;

		_jacobianOplusXj(1, 0) = z;
		_jacobianOplusXj(1, 1) = 0;
		_jacobianOplusXj(1, 2) = -x;
		_jacobianOplusXj(1, 3) = 0;
		_jacobianOplusXj(1, 4) = -1;
		_jacobianOplusXj(1, 5) = 0;

		_jacobianOplusXj(2, 0) = -y;
		_jacobianOplusXj(2, 1) = x;
		_jacobianOplusXj(2, 2) = 0;
		_jacobianOplusXj(2, 3) = 0;
		_jacobianOplusXj(2, 4) = 0;
		_jacobianOplusXj(2, 5) = -1;
	}

	void EdgeProjectXYZRGBDPoseOnly::computeError()
	{
		const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
		_error = _measurement - pose->estimate().map(point_);
	}

	void EdgeProjectXYZRGBDPoseOnly::linearizeOplus()
	{
		g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]);
		g2o::SE3Quat T(pose->estimate());
		Eigen::Vector3d xyz_trans = T.map(point_);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];

		_jacobianOplusXi(0, 0) = 0;
		_jacobianOplusXi(0, 1) = -z;
		_jacobianOplusXi(0, 2) = y;
		_jacobianOplusXi(0, 3) = -1;
		_jacobianOplusXi(0, 4) = 0;
		_jacobianOplusXi(0, 5) = 0;

		_jacobianOplusXi(1, 0) = z;
		_jacobianOplusXi(1, 1) = 0;
		_jacobianOplusXi(1, 2) = -x;
		_jacobianOplusXi(1, 3) = 0;
		_jacobianOplusXi(1, 4) = -1;
		_jacobianOplusXi(1, 5) = 0;

		_jacobianOplusXi(2, 0) = -y;
		_jacobianOplusXi(2, 1) = x;
		_jacobianOplusXi(2, 2) = 0;
		_jacobianOplusXi(2, 3) = 0;
		_jacobianOplusXi(2, 4) = 0;
		_jacobianOplusXi(2, 5) = -1;
	}

	void EdgeProjectXYZ2UVPoseOnly::computeError()
	{
		const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
		_error = _measurement - convert_->toVector2d(camera_->camera2pixel(convert_->toPoint3f(pose->estimate().map(point_))));
	}

	void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
	{
		g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]);
		g2o::SE3Quat T(pose->estimate());
		Eigen::Vector3d xyz_trans = T.map(point_);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];
		double z_2 = z*z;

		_jacobianOplusXi(0, 0) = x*y / z_2 *camera_->fx_;
		_jacobianOplusXi(0, 1) = -(1 + (x*x / z_2)) *camera_->fx_;
		_jacobianOplusXi(0, 2) = y / z * camera_->fx_;
		_jacobianOplusXi(0, 3) = -1. / z * camera_->fx_;
		_jacobianOplusXi(0, 4) = 0;
		_jacobianOplusXi(0, 5) = x / z_2 * camera_->fx_;

		_jacobianOplusXi(1, 0) = (1 + y*y / z_2) *camera_->fy_;
		_jacobianOplusXi(1, 1) = -x*y / z_2 *camera_->fy_;
		_jacobianOplusXi(1, 2) = -x / z *camera_->fy_;
		_jacobianOplusXi(1, 3) = 0;
		_jacobianOplusXi(1, 4) = -1. / z *camera_->fy_;
		_jacobianOplusXi(1, 5) = y / z_2 *camera_->fy_;
	}

	void Optimizer::LocalBundleAdjustment(Frame* curr_, std::vector<int> inliers, cv::Mat& T_c_r_estimate)
	{
		// using bundle adjustment to optimize the pose 
		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
		Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
		Block* solver_ptr = new Block(linearSolver);
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		g2o::SparseOptimizer optimizer;
		optimizer.setAlgorithm(solver);

		g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
		pose->setId(0);
		pose->setEstimate(g2o::SE3Quat(convert_->toSE3Quat(T_c_r_estimate)));
		optimizer.addVertex(pose);

		// edges
		for (int i = 0; i < inliers.size(); i++)
		{
			int index = inliers[i];
			// 3D -> 2D projection
			EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
			edge->setId(i);
			edge->setVertex(0, pose);
			edge->camera_ = curr_->camera_;
			edge->point_ = Eigen::Vector3d(curr_->m_vpts3d[index].x, curr_->m_vpts3d[index].y, curr_->m_vpts3d[index].z);
			edge->setMeasurement(Eigen::Vector2d(curr_->m_vpts2d[index].x, curr_->m_vpts2d[index].y));
			edge->setInformation(Eigen::Matrix2d::Identity());
			optimizer.addEdge(edge);
		}

		optimizer.initializeOptimization();
		optimizer.optimize(10);

		T_c_r_estimate = convert_->toCvSE3(pose->estimate().rotation().matrix(), Eigen::Vector3d(pose->estimate().translation()));
	}

	void Optimizer::PoseOptimization(Frame* curr_, std::vector<int> inliers, cv::Mat& T_c_r_estimate)
	{
		// 构造g2o优化器
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
		pose->setId(0);
		pose->setEstimate(g2o::SE3Quat(convert_->toSE3Quat(T_c_r_estimate)));
		pose->setFixed(false);
		optimizer.addVertex(pose);

		// edges
		for (int i = 0; i < inliers.size(); i++)
		{
			int index = inliers[i];
			// 3D -> 2D projection
			g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
			e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
			e->setId(i);

			Eigen::Matrix<double, 2, 1> obs;
			obs << curr_->m_vpts2d[index].x, curr_->m_vpts2d[index].y;
			e->setMeasurement(obs);

			//加了这个核函数,反而不太平滑了？
			//g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			//e->setRobustKernel(rk);
			e->setInformation(Eigen::Matrix2d::Identity());

			e->fx = curr_->camera_->fx_;
			e->fy = curr_->camera_->fy_;
			e->cx = curr_->camera_->cx_;
			e->cy = curr_->camera_->cy_;

			e->Xw[0] = curr_->m_vpts3d[index].x;
			e->Xw[1] = curr_->m_vpts3d[index].y;
			e->Xw[2] = curr_->m_vpts3d[index].z;

			optimizer.addEdge(e);
		}

		optimizer.initializeOptimization(0);
		optimizer.optimize(10);

		T_c_r_estimate = convert_->toCvSE3(pose->estimate().rotation().matrix(), Eigen::Vector3d(pose->estimate().translation()));
	}
}
#include "viewer.h"

namespace myslam
{
	Viewer::Viewer(std::string strSettingPath)
	{
		cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

		float fps = fSettings["Camera.fps"];
		if (fps < 1)
			fps = 30;
		mT = 1e3 / fps;

		mImageWidth = fSettings["Camera.width"];
		mImageHeight = fSettings["Camera.height"];
		if (mImageWidth < 1 || mImageHeight < 1)
		{
			mImageWidth = 640;
			mImageHeight = 480;
		}

		mViewpointX = fSettings["Viewer.ViewpointX"];
		mViewpointY = fSettings["Viewer.ViewpointY"];
		mViewpointZ = fSettings["Viewer.ViewpointZ"];
		mViewpointF = fSettings["Viewer.ViewpointF"];

		mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
		mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
		mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
		mPointSize = fSettings["Viewer.PointSize"];
		mCameraSize = fSettings["Viewer.CameraSize"];
		mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

		VisualOdometry_ = nullptr;
	}

	// pangolin库的文档：http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html
	void Viewer::Run()
	{
		{
			//互斥做,初始化显示窗口时,不进行位姿运算,对应visual_odometry中的addframe函数
			std::unique_lock<std::mutex> lock(mMutexViewer);
			pangolin::CreateWindowAndBind("my-slam: Map Viewer", 1024, 768);

			// 启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
			glEnable(GL_DEPTH_TEST);

			// 在OpenGL中使用颜色混合
			glEnable(GL_BLEND);

			// 选择混合选项
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			// 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
			pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
			pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
			pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
			pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
			pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
			pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
			pangolin::Var<bool> menuReset("menu.Reset", false, false);

			// 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
			// 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
			//                观测目标位置：(0, 0, 0)
			//                观测的方位向量：(0.0,-1.0, 0.0)
			s_cam = pangolin::OpenGlRenderState(
				pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
				pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
			);

			Tcw.SetIdentity();
			cv::namedWindow("my-slam: Current Frame");

			// 定义显示面板大小，orbslam中有左右两个面板，左边显示一些按钮，右边显示图形
			// 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
			// 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
			// 最后一个参数（-1024.0f/768.0f）为显示长宽比
			d_cam = pangolin::CreateDisplay()
				.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
				.SetHandler(new pangolin::Handler3D(s_cam));
		}

		while (1)
		{
			// 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// 得到最新的相机位姿
			GetCurrentOpenGLCameraMatrix(Tcw);

			s_cam.Follow(Tcw);
			d_cam.Activate(s_cam);

			// 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			DrawCurrentCamera(Tcw);
			DrawTrackingFrame();
			DrawMapPoints();

			pangolin::FinishFrame();

			cv::Mat im = DrawFrame();
			if (im.empty())
				continue;

			cv::imshow("my-slam: Current Frame", im);
			cv::waitKey(10);
		}
	}

	void Viewer::DrawTrackingFrame()
	{
		const float &w = mKeyFrameSize*0.3;
		const float h = w*0.75;
		const float z = w*0.6;

		{
			//互斥锁,在使用关键帧信息时,不能修改
			std::unique_lock<std::mutex> lock(mMutexTrackingFrame);

			//绘制所有的关键帧位置
			for (int i = 0; i < mvAllFrame.size(); i++)
			{
				Frame* cur = mvAllFrame[i];
				cv::Mat Twc = cur->T_c_w_.inv();

				pangolin::OpenGlMatrix M;
				if (!Twc.empty())
				{
					cv::Mat Rwc(3, 3, CV_32F);
					cv::Mat twc(3, 1, CV_32F);
					{
						Rwc = Twc.rowRange(0, 3).colRange(0, 3);
						twc = Twc.rowRange(0, 3).col(3);
					}

					M.m[0] = Rwc.at<float>(0, 0);
					M.m[1] = Rwc.at<float>(1, 0);
					M.m[2] = Rwc.at<float>(2, 0);
					M.m[3] = 0.0;

					M.m[4] = Rwc.at<float>(0, 1);
					M.m[5] = Rwc.at<float>(1, 1);
					M.m[6] = Rwc.at<float>(2, 1);
					M.m[7] = 0.0;

					M.m[8] = Rwc.at<float>(0, 2);
					M.m[9] = Rwc.at<float>(1, 2);
					M.m[10] = Rwc.at<float>(2, 2);
					M.m[11] = 0.0;

					M.m[12] = twc.at<float>(0);
					M.m[13] = twc.at<float>(1);
					M.m[14] = twc.at<float>(2);
					M.m[15] = 1.0;
				}
				else
					M.SetIdentity();
				
				glPushMatrix();
				glMultMatrixd(M.m);

				glLineWidth(mKeyFrameLineWidth);
				glColor3f(0.0f, 0.0f, 1.0f);
				glBegin(GL_LINES);
				glVertex3f(0, 0, 0);
				glVertex3f(w, h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(w, -h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(-w, -h, z);
				glVertex3f(0, 0, 0);
				glVertex3f(-w, h, z);

				glVertex3f(w, h, z);
				glVertex3f(w, -h, z);

				glVertex3f(-w, h, z);
				glVertex3f(-w, -h, z);

				glVertex3f(-w, h, z);
				glVertex3f(w, h, z);

				glVertex3f(-w, -h, z);
				glVertex3f(w, -h, z);
				glEnd();

				glPopMatrix();
			}
		}
	}

	void Viewer::DrawMapPoints()
	{
		{
			//互斥锁,在使用三维点信息时,不能修改
			std::unique_lock<std::mutex> lock(mMutex3dPoints);
			glPointSize(mPointSize);
			glBegin(GL_POINTS);
			glColor3f(0.0, 0.0, 0.0);

			//绘制所有的三维点
			for (int i = 0; i < mAll3dpts.size(); i+=10)
			{
				cv::Mat pos = cv::Mat(mAll3dpts[i]);
				glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
			}
			glEnd();
		}
	}

	cv::Mat Viewer::DrawFrame()
	{
		const float r = 5;

		if (VisualOdometry_ == nullptr)
			return cv::Mat();

		if (VisualOdometry_->keypoints_curr_.size() == 0 || VisualOdometry_->feature_matches_.size() == 0)
			return cv::Mat();

		{
			//互斥锁,在使用里程计信息时,不能修改
			std::unique_lock<std::mutex> lock(mMutex);

			mkps = VisualOdometry_->keypoints_curr_;
			mfeature_matches = VisualOdometry_->feature_matches_;
			mIm = VisualOdometry_->curr_->color_;

			n = mfeature_matches.size();
		}

		//在图像上绘制出二维点坐标
		for (int i = 0; i < n; i++)
		{
			int index = mfeature_matches[i].trainIdx;
			cv::Point2f pt1, pt2;
			pt1.x = mkps[index].pt.x - r;
			pt1.y = mkps[index].pt.y - r;
			pt2.x = mkps[index].pt.x + r;
			pt2.y = mkps[index].pt.y + r;

			cv::rectangle(mIm, pt1, pt2, cv::Scalar(0, 255, 0));
			cv::circle(mIm, mkps[index].pt, 2, cv::Scalar(0, 255, 0), -1);
		}

		return mIm;
	}

	void Viewer::SetCurrentCameraPose(const cv::Mat &Tcw)
	{
		//互斥锁,在使用相机位姿时,不能修改其值
		std::unique_lock<std::mutex> lock(mMutexCamera);
		mCameraPose = Tcw.clone();
		mCameraPose = mCameraPose.inv();
	}

	void Viewer::SetVisualOdometry(myslam::VisualOdometry* pVisualOdometry)
	{
		//互斥锁,在使用里程计信息时,不能修改
		std::unique_lock<std::mutex> lock(mMutex);
		VisualOdometry_ = pVisualOdometry;
	}

	void Viewer::GetAllFrame(std::vector<Frame*> vAllFrame)
	{
		//互斥锁,在使用关键帧信息时,不能修改
		std::unique_lock<std::mutex> lock(mMutexTrackingFrame);
		mvAllFrame = vAllFrame;
	}

	void Viewer::GetAll3dPoints(std::vector<cv::Point3f> All3dpts)
	{
		//互斥锁,在使用三维点信息时,不能修改
		std::unique_lock<std::mutex> lock(mMutex3dPoints);
		mAll3dpts = All3dpts;
	}

	//关于gl相关的函数，可直接google, 并加上msdn关键词
	void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
	{
		//相机模型大小
		const float &w = mCameraSize*0.5;
		const float h = w * 0.75;
		const float z = w * 0.6;

		glPushMatrix();

		//将4*4的矩阵Twc.m右乘一个当前矩阵
		//（由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
		//因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
		glMultMatrixd(Twc.m);

		//设置绘制图形时线的宽度
		glLineWidth(mCameraLineWidth);

		//设置当前颜色为绿色(相机图标显示为绿色)
		glColor3f(0.0f, 1.0f, 0.0f);

		//用线将下面的顶点两两相连
		glBegin(GL_LINES);
		glVertex3f(0, 0, 0);
		glVertex3f(w, h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(w, -h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(-w, -h, z);
		glVertex3f(0, 0, 0);
		glVertex3f(-w, h, z);

		glVertex3f(w, h, z);
		glVertex3f(w, -h, z);

		glVertex3f(-w, h, z);
		glVertex3f(-w, -h, z);

		glVertex3f(-w, h, z);
		glVertex3f(w, h, z);

		glVertex3f(-w, -h, z);
		glVertex3f(w, -h, z);
		glEnd();

		glPopMatrix();
	}

	// 将相机位姿mCameraPose由Mat类型转化为OpenGlMatrix类型
	void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
	{
		if (!mCameraPose.empty())
		{
			cv::Mat Rcw(3, 3, CV_32F);
			cv::Mat tcw(3, 1, CV_32F);
			{
				//互斥锁,在使用相机位姿时,不能修改其值
				std::unique_lock<std::mutex> lock(mMutexCamera);
				Rcw = mCameraPose.rowRange(0, 3).colRange(0, 3);
				tcw = mCameraPose.rowRange(0, 3).col(3);
			}

			M.m[0] = Rcw.at<float>(0, 0);
			M.m[1] = Rcw.at<float>(1, 0);
			M.m[2] = Rcw.at<float>(2, 0);
			M.m[3] = 0.0;

			M.m[4] = Rcw.at<float>(0, 1);
			M.m[5] = Rcw.at<float>(1, 1);
			M.m[6] = Rcw.at<float>(2, 1);
			M.m[7] = 0.0;

			M.m[8] = Rcw.at<float>(0, 2);
			M.m[9] = Rcw.at<float>(1, 2);
			M.m[10] = Rcw.at<float>(2, 2);
			M.m[11] = 0.0;

			M.m[12] = tcw.at<float>(0);
			M.m[13] = tcw.at<float>(1);
			M.m[14] = tcw.at<float>(2);
			M.m[15] = 1.0;
		}
		else
			M.SetIdentity();
	}
}

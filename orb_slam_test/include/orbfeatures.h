#ifndef ORBFEATURES_H
#define ORBFEATURES_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
	class ExtractorNode
	{
	public:
		ExtractorNode() :bNoMore(false) {}

		void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

		std::vector<cv::KeyPoint> vKeys;
		cv::Point2i UL, UR, BL, BR;
		std::list<ExtractorNode>::iterator lit;
		bool bNoMore;
	};

	class ORBextractor
	{
	public:
		enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

		ORBextractor(int _nfeatures = 1000, float _scaleFactor = 1.2, int _nlevels = 8, int _iniThFAST = 20, int _minThFAST = 12);
		~ORBextractor() {}

		// Compute the ORB features and descriptors on an image.
		// ORB are dispersed on the image using an octree.
		// Mask is ignored in the current implementation.
		bool runORBextractor(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

		int inline GetLevels() {
			return nlevels;
		}

		float inline GetScaleFactor() {
			return scaleFactor;
		}

		std::vector<float> inline GetScaleFactors() {
			return mvScaleFactor;
		}

		std::vector<float> inline GetInverseScaleFactors() {
			return mvInvScaleFactor;
		}

		std::vector<float> inline GetScaleSigmaSquares() {
			return mvLevelSigma2;
		}

		std::vector<float> inline GetInverseScaleSigmaSquares() {
			return mvInvLevelSigma2;
		}

		std::vector<cv::Mat> mvImagePyramid;

	private:
		float IC_Angle(cv::Mat image, cv::Point2f pt, std::vector<int> u_max);
		void computeDescriptors(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat& descriptors, std::vector<cv::Point> pattern);
		void computeOrbDescriptor(cv::KeyPoint kpt, cv::Mat img, cv::Point* pattern, uchar* desc);
		void computeOrientation(cv::Mat image, std::vector<cv::KeyPoint> keypoints, std::vector<int> umax);

		void ComputePyramid(cv::Mat image);
		void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

		std::vector<cv::KeyPoint> DistributeOctTree(std::vector<cv::KeyPoint> vToDistributeKeys, const int minX,
			const int maxX, const int minY, const int maxY, const int N);

		int nfeatures;
		double scaleFactor;
		int nlevels;
		int iniThFAST;
		int minThFAST;

		std::vector<cv::Point> pattern;

		std::vector<int> mnFeaturesPerLevel;
		std::vector<int> umax;

		std::vector<float> mvScaleFactor;
		std::vector<float> mvInvScaleFactor;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
	};
}

#endif
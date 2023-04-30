#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include <opencv2/opencv.hpp>

#define THRESH_FACTOR 6

using namespace cv;
using namespace std;

// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };


class Top_FeatureMatcher
{
public:
	Top_FeatureMatcher() {};
	// OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches 
	Top_FeatureMatcher(const vector<KeyPoint> vkp1, const Size size1, const vector<KeyPoint> vkp2, const Size size2, const vector<DMatch> &vDMatches)
	{
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
		mGridSizeLeft = Size(20, 20);
		// 左图像网格的数量：400
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neihbor of left grid 400*9
		mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
		InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
	};
	~Top_FeatureMatcher() {};


private:

	// Normalized Points
	vector<Point2f> mvP1, mvP2;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	Size mGridSizeLeft, mGridSizeRight;
	int mGridNumberLeft;
	int mGridNumberRight;


	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	Mat mMotionStatistics;

	// 左图像每个网格的初始匹配的特征数量
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	//
	Mat mGridNeighborLeft;
	Mat mGridNeighborRight;


public:

	// Matches
	vector<pair<int, int> > mvMatches;

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

	bool gms_init(const vector<KeyPoint> vkp1, const Size size1, const vector<KeyPoint> vkp2, const Size size2, const vector<int> &vnMatches12);

	void imresize(Mat src, int height);

	Mat DrawInlier(Mat src1, Mat src2, vector<KeyPoint> kpt1, vector<KeyPoint> kpt2, vector<pair<int, int> > inlier, int type);
	
	Mat DrawInlier(Mat src1, Mat src2, vector<KeyPoint> kpt1, vector<KeyPoint> kpt2, vector<DMatch> inlier, int type);

private:

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const vector<KeyPoint> kp, const Size size, vector<Point2f>& npts);
	// Convert OpenCV DMatch to Match (pair<int, int>)
	// 把opencv中的DMatch类型转换成两组特征点匹配的特征点序列号
	void ConvertMatches(const vector<DMatch> vDMatches, vector<pair<int, int> > &vMatches);
	void ConvertMatches(const vector<int> vnMatches12);

	int GetGridIndexLeft(const Point2f pt, int type);

	int GetGridIndexRight(const Point2f pt);

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	// GetNB9就是获取一个格子它周围的九宫格
	vector<int> GetNB9(const int idx, const Size GridSize);

	//400*9 20*20
	void InitalizeNiehbors(Mat neighbor, const Size GridSize);

	void SetScale(int Scale);

	// Run 
	int run(int RotationType);
};
#endif
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main() {

	Mat img1 = imread("featurematching1.jpg", 0);
	Mat img2 = imread("featurematching2.jpg", 0);
	Mat features1 = imread("featurematching1.jpg", 1);
	Mat features2 = imread("featurematching2.jpg", 1);
	Mat descriptor1;
	Mat descriptor2;
	Mat result;

	vector<KeyPoint> keypoints1, keypoints2;

	resize(img1, img1, Size(360, 480));
	resize(img2, img2, Size(360, 480));
	resize(features1, features1, Size(360, 480));
	resize(features2, features2, Size(360, 480));

	Ptr<FeatureDetector> detector = ORB::create(1000);

	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	drawKeypoints(features1, keypoints1, features1);
	drawKeypoints(features2, keypoints2, features2);

	Ptr<DescriptorExtractor> extractor = ORB::create(1000);
	extractor->compute(img1, keypoints1, descriptor1);
	extractor->compute(img2, keypoints2, descriptor2);

	descriptor1.convertTo(descriptor1, CV_32F);
	descriptor2.convertTo(descriptor2, CV_32F);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
	float ratio_threshold = 0.7;

	matcher->knnMatch(descriptor1, descriptor2, matches, 2); // 2 as parameter to get best two matches

	for(int i=0; i<matches.size(); i++) {
		if(matches[i][0].distance < matches[i][1].distance * ratio_threshold) {
			good_matches.push_back(matches[i][0]);
		}
	}

	drawMatches(features1, keypoints1, features2, keypoints2, good_matches, result);

	namedWindow("Input 1", WINDOW_NORMAL);
	namedWindow("Input 2", WINDOW_NORMAL);
	namedWindow("Features 1", WINDOW_NORMAL);
	namedWindow("Features 2", WINDOW_NORMAL);
	namedWindow("result", WINDOW_NORMAL);

	imshow("Input 1", img1);
	imshow("Input 2", img2);
	imshow("Features 1", features1);
	imshow("Features 2", features2);
	imshow("result", result);

	waitKey(0);

	return 0;
}
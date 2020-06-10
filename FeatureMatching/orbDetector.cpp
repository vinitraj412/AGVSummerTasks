#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main() {

	Mat input_img = imread("featurematching1.jpg", 0);
	Mat result = imread("featurematching1.jpg", 1);

	resize(input_img, input_img, Size(360, 480));
	resize(result, result, Size(360, 480));

	vector<KeyPoint> keypoints;
	vector<Point2f> keypoint_points;

	Ptr<FeatureDetector> detector = ORB::create(1000);

	detector->detect(input_img, keypoints);

	KeyPoint::convert(keypoints, keypoint_points);

	for(int i=0; i<keypoint_points.size(); i++) {
		cout << keypoint_points[i].x << ", " << keypoint_points[i].y << endl;
		//circle(result, keypoint_points[i], 5, Scalar(0, 0, 255), 2);
	}

	drawKeypoints(result, keypoints, result);

	namedWindow("Input Image", WINDOW_NORMAL);
	namedWindow("Keypoints", WINDOW_NORMAL);
	imshow("Input Image", input_img);
	imshow("Keypoints", result);

	waitKey(0);

	return 0;
}
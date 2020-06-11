#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace std;
using namespace cv;

class ShadowDetection {

	Mat img;

public:
	ShadowDetection(Mat input_img) {
		img = input_img;
	}

	Mat convert_to_ycrcb() {

		Mat result;
		cvtColor(img, result, CV_BGR2YCrCb);

		return result;
	}

	Mat get_Y_channel(Mat ycrcb) {

		Mat result;
		Mat channels[3];

		split(ycrcb, channels);

		result = channels[0];
		return result;

	}

	double calculate_mean_Y(Mat y_channel) {
		double mean_Y = 0.0;

		for(int i=0; i<y_channel.rows; i++) {
			for(int j=0; j<y_channel.cols; j++) {
				mean_Y += y_channel.at<uchar>(i, j);
			}
		}

		mean_Y /= (y_channel.rows) * (y_channel.cols);

		return mean_Y;
	}

	double calculate_standard_deviation(Mat y_channel, double mean_Y) {
		double sd = 0;

		for(int i=0; i<y_channel.rows; i++) {
			for(int j=0; j<y_channel.cols; j++) {
				sd += pow(mean_Y - y_channel.at<uchar>(i, j), 2);
			}
		}

		sd /= (y_channel.rows) * (y_channel.cols);
		sd = sqrt(sd);

		return sd;
	}

	Mat detect() {

		Mat ycrcb = convert_to_ycrcb();
		Mat y_channel_1 = get_Y_channel(ycrcb);
		Mat y_channel;

		equalizeHist(y_channel_1, y_channel);

		double mean_Y = calculate_mean_Y(y_channel);
		double sd = calculate_standard_deviation(y_channel, mean_Y);

		cout << "Mean Y: " << mean_Y << endl;
		cout << "Standard Deviation: " << sd << endl;

		Mat detect_1 = Mat(img.rows, img.cols, CV_8UC1, Scalar(0));

		for(int i=0; i<y_channel.rows; i++) {
			for(int j=0; j<y_channel.cols; j++) {
				if(y_channel.at<uchar>(i, j) < sd) {
					detect_1.at<uchar>(i, j) = 255;
				}
			}
		}

		namedWindow("Detect1", WINDOW_NORMAL);
		imshow("Detect1", detect_1);

		Mat detect_2 = detect_1.clone();
		

		for(int i=0; i<y_channel.rows; i++) {
			for(int j=0; j<y_channel.cols; j++) {

				if(detect_1.at<uchar>(i, j) == 0) {

					//Mat temp = Mat(3, 3, CV_8UC3, Scalar(0, 0, 0));
				//Mat window;
				//cvtColor(temp, window, COLOR_BGR2y_channel);

				vector<int> values;

				for(int i1=-1; i1<=1; i1++) {
					for(int j1=-1; j1<=1; j1++) {
						int i2 = i + i1, j2 = j + j1;

						if(i2 >= 0 && i2 < y_channel.rows && j2 >= 0 && j2 < y_channel.cols) {

							//if(detect_1.at<uchar>(i2, j2) == 0) {

								values.push_back(y_channel.at<uchar>(i2, j2));
								//window.at<Vec3b>(i1 + 1, j1 + 1)[0] = y_channel.at<Vec3b>(i2, j2)[0];
								//window.at<Vec3b>(i1 + 1, j1 + 1)[1] = y_channel.at<Vec3b>(i2, j2)[1];
								//window.at<Vec3b>(i1 + 1, j1 + 1)[2] = y_channel.at<Vec3b>(i2, j2)[2];

							//}
						}
						else {
							values.push_back(0);
						}
					}
				}

				//double window_mean = calculate_mean_Y(window);
				//double window_sd = calculate_standard_deviation(window, window_mean);

				double window_mean=0.0, window_sd = 0.0;

				for(int k=0; k<values.size(); k++) {
					window_mean += values.at(k);
				}

				if(values.size() != 0) {
					window_mean = window_mean / values.size();
					
					for(int k=0; k<values.size(); k++) {
						window_sd += pow(values.at(k) - window_mean, 2);
					}

					window_sd = window_sd / values.size();
					window_sd = sqrt(window_sd);

					if(y_channel.at<uchar>(i, j) < window_sd) {
						detect_2.at<uchar>(i, j) = 255;
					}
					/*
					else {
						detect_2.at<uchar>(i, j) = 0;
					}
					*/
				}

				//cout << "WINDOW MEAN: " << window_mean << ", ";
				//cout << "WINDOW_SD: " << window_sd << endl;
				/*
				if(ycrcb.at<Vec3b>(i, j)[0] < window_sd) {
					detect_2.at<uchar>(i, j) = 255;
				}
				*/

				}

				
			}
		}

		namedWindow("Detect2", WINDOW_NORMAL);
		imshow("Detect2", detect_2);

		Mat result = detect_2.clone();

		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(result, result, kernel);
		dilate(result, result, kernel);

		return result;

	}
};

class shadow_removal
{
	Mat mask;
	Mat img;
public:
	shadow_removal(Mat inpt_img, Mat inpt_mask){
		mask=inpt_mask;
		img=inpt_img;
	}

	Mat convert_to_ycrcb(){

		Mat result;
		cvtColor(img, result, CV_BGR2YCrCb);

		return result;
	}
	double getavg(Mat ycrcb, int k){
		int counter=0;
		double avg=0;
		for(int i=0; i<img.rows; i++){
			for(int j=0; j<img.cols; j++){
				if(k==0){
					if(mask.at<uchar>(i,j)==255){
						avg+=ycrcb.at<Vec3b>(i,j)[0];
						counter++;
					}
				}
				else{
					if(mask.at<uchar>(i,j)==0){
						avg+=ycrcb.at<Vec3b>(i,j)[0];
						counter++;
					}
				}
			}
		}
		avg=avg/counter;
		return avg;
	}
	void add_val(int diff, Mat &intensity_correction){
		for(int i=0; i<img.rows; i++){
			for(int j=0; j<img.cols; j++){
				if(mask.at<uchar>(i,j)==255){
					intensity_correction.at<Vec3b>(i,j)[0]+=diff;
				}
			}
		}
	}
	Mat remove(){
		Mat ycrcb=convert_to_ycrcb();
		double avg_shadow=getavg(ycrcb, 0);
		double avg_light=getavg(ycrcb, 1);
		int diff=avg_light - avg_shadow;
		double r=avg_light/avg_shadow;
		cout<<"diff is "<<diff<<endl;
		Mat intensity_correction=convert_to_ycrcb();
		add_val(diff, intensity_correction);
		cvtColor(intensity_correction, intensity_correction, CV_YCrCb2BGR);
		return intensity_correction;
	}
	
};

int main() {

	Mat img = imread("shadow_2.jpg", 1);
	Mat ycrcb, result, intensity_correction;
	Mat channels[3];
	Mat y_ch, y_ch_2;

	ShadowDetection detector(img);
	ycrcb = detector.convert_to_ycrcb();
	result = detector.detect();

	shadow_removal remover(img, result);
	intensity_correction=remover.remove();

	split(ycrcb, channels);
	y_ch = channels[0];

	equalizeHist(y_ch, y_ch_2);

	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("YCrCb", WINDOW_NORMAL);
	namedWindow("Y", WINDOW_NORMAL);
	namedWindow("Y_2", WINDOW_NORMAL);
	namedWindow("Result", WINDOW_NORMAL);
	namedWindow("Intensity_correction", WINDOW_NORMAL);

	imshow("Input", img);
	imshow("YCrCb", ycrcb);
	imshow("Y", y_ch);
	imshow("Y_2", y_ch_2);
	imshow("Result", result);
	imshow("Intensity_correction", intensity_correction);

	waitKey(0);

	return 0;
}
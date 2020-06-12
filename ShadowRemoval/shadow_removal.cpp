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

					if(intensity_correction.at<Vec3b>(i, j)[0] + diff < 255) {
						intensity_correction.at<Vec3b>(i,j)[0]+=diff;	
					}
					else {
						intensity_correction.at<Vec3b>(i,j)[0] = 255;
					}
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

		Mat channels[3];
		split(ycrcb, channels);
		Mat y_channel = channels[0];

		Mat y_channel_equ;
		equalizeHist(y_channel, y_channel_equ);

		int y_max = 0, y_min = 300;

		for(int i=0; i<y_channel_equ.rows; i++) {
			for(int j=0; j<y_channel_equ.cols; j++) {
				if(mask.at<uchar>(i, j) == 255) {
					if(y_max < y_channel_equ.at<uchar>(i, j)) {
						y_max = y_channel_equ.at<uchar>(i, j);
					}

					if(y_min > y_channel_equ.at<uchar>(i, j)) {
						y_min = y_channel_equ.at<uchar>(i, j);	
					}
				}
			}
		}

		cout << "Y_MAX: " << y_max << ", Y_MIN: " << y_min << endl;

		Mat umbra = Mat(mask.rows, mask.cols, CV_8UC1, Scalar(0));
		Mat penumbra = umbra.clone(), sunlight= umbra.clone();

		for(int i=0; i<y_channel_equ.rows; i++) {
			for(int j=0; j<y_channel_equ.cols; j++) {
				if(mask.at<uchar>(i, j) == 255) {
					if(((double) y_max - y_channel_equ.at<uchar>(i, j)) / (y_max - y_min) <= 0.25) {
						sunlight.at<uchar>(i, j) = 255;
					}
					else if(((double) y_max - y_channel_equ.at<uchar>(i, j)) / (y_max - y_min) >= 0.75) {
						umbra.at<uchar>(i, j) = 255;
					}
					else {
						penumbra.at<uchar>(i, j) = 255;
					}
				}
			}
		}

		double avg_non_shadow_r = 0.0, avg_non_shadow_b = 0.0, avg_non_shadow_g = 0.0;

		int counter = 0;

		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(mask.at<uchar>(i, j) == 0) {
					counter++;
					avg_non_shadow_b += intensity_correction.at<Vec3b>(i, j)[0];
					avg_non_shadow_g += intensity_correction.at<Vec3b>(i, j)[1];
					avg_non_shadow_r += intensity_correction.at<Vec3b>(i, j)[2];
				}
			}
		}

		avg_non_shadow_b /= counter;
		avg_non_shadow_g /= counter;
		avg_non_shadow_r /= counter;

		Mat color_correction = intensity_correction.clone();

		cout << "AVG_B: " << avg_non_shadow_b << ", AVG_G: " << avg_non_shadow_g << ", AVG_R: " << avg_non_shadow_r << endl;

		double const_b, const_r, const_g;

		double avg_b, avg_g, avg_r;
		
		counter = 0;
		avg_b = 0.0;
		avg_g = 0.0;
		avg_r = 0.0;
		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(penumbra.at<uchar>(i, j) == 255) {
					counter++;
					avg_b += intensity_correction.at<Vec3b>(i, j)[0];
					avg_g += intensity_correction.at<Vec3b>(i, j)[1];
					avg_r += intensity_correction.at<Vec3b>(i, j)[2];
				}
			}
		}

		avg_b /= counter;
		avg_g /= counter;
		avg_r /= counter;

		const_b = avg_non_shadow_b / avg_b;
		const_g = avg_non_shadow_g / avg_g;
		const_r = avg_non_shadow_r / avg_r;

		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(penumbra.at<uchar>(i, j) == 255) {

					int b, g, r;

					b = (intensity_correction.at<Vec3b>(i, j)[0] * const_b + 0.5);
					g = (intensity_correction.at<Vec3b>(i, j)[1] * const_g + 0.5);
					r = (intensity_correction.at<Vec3b>(i, j)[2] * const_r + 0.5);

					if(b < 255)
						color_correction.at<Vec3b>(i, j)[0] = b;
					else
						color_correction.at<Vec3b>(i, j)[0] = 255;
					if(g < 255)
						color_correction.at<Vec3b>(i, j)[1] = g;
					else
						color_correction.at<Vec3b>(i, j)[1] = 255;
					if(r < 255)
						color_correction.at<Vec3b>(i, j)[2] = r;
					else
						color_correction.at<Vec3b>(i, j)[2] = 255;
				}
			}
		}

		counter = 0;
		avg_b = 0.0;
		avg_g = 0.0;
		avg_r = 0.0;
		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(umbra.at<uchar>(i, j) == 255) {
					counter++;
					avg_b += intensity_correction.at<Vec3b>(i, j)[0];
					avg_g += intensity_correction.at<Vec3b>(i, j)[1];
					avg_r += intensity_correction.at<Vec3b>(i, j)[2];
				}
			}
		}

		avg_b /= counter;
		avg_g /= counter;
		avg_r /= counter;

		const_b = avg_non_shadow_b / avg_b;
		const_g = avg_non_shadow_g / avg_g;
		const_r = avg_non_shadow_r / avg_r;

		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(umbra.at<uchar>(i, j) == 255) {

					int b, g, r;

					b = (intensity_correction.at<Vec3b>(i, j)[0] * const_b + 0.5);
					g = (intensity_correction.at<Vec3b>(i, j)[1] * const_g + 0.5);
					r = (intensity_correction.at<Vec3b>(i, j)[2] * const_r + 0.5);

					if(b < 255)
						color_correction.at<Vec3b>(i, j)[0] = b;
					else
						color_correction.at<Vec3b>(i, j)[0] = 255;
					if(g < 255)
						color_correction.at<Vec3b>(i, j)[1] = g;
					else
						color_correction.at<Vec3b>(i, j)[1] = 255;
					if(r < 255)
						color_correction.at<Vec3b>(i, j)[2] = r;
					else
						color_correction.at<Vec3b>(i, j)[2] = 255;
				}
			}
		}

		counter = 0;
		avg_b = 0.0;
		avg_g = 0.0;
		avg_r = 0.0;
		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(sunlight.at<uchar>(i, j) == 255) {
					counter++;
					avg_b += intensity_correction.at<Vec3b>(i, j)[0];
					avg_g += intensity_correction.at<Vec3b>(i, j)[1];
					avg_r += intensity_correction.at<Vec3b>(i, j)[2];
				}
			}
		}

		avg_b /= counter;
		avg_g /= counter;
		avg_r /= counter;

		const_b = avg_non_shadow_b / avg_b;
		const_g = avg_non_shadow_g / avg_g;
		const_r = avg_non_shadow_r / avg_r;

		for(int i=0; i<intensity_correction.rows; i++) {
			for(int j=0; j<intensity_correction.cols; j++) {
				if(sunlight.at<uchar>(i, j) == 255) {

					int b, g, r;

					b = (intensity_correction.at<Vec3b>(i, j)[0] * const_b + 0.5);
					g = (intensity_correction.at<Vec3b>(i, j)[1] * const_g + 0.5);
					r = (intensity_correction.at<Vec3b>(i, j)[2] * const_r + 0.5);

					if(b < 255)
						color_correction.at<Vec3b>(i, j)[0] = b;
					else
						color_correction.at<Vec3b>(i, j)[0] = 255;
					if(g < 255)
						color_correction.at<Vec3b>(i, j)[1] = g;
					else
						color_correction.at<Vec3b>(i, j)[1] = 255;
					if(r < 255)
						color_correction.at<Vec3b>(i, j)[2] = r;
					else
						color_correction.at<Vec3b>(i, j)[2] = 255;
				}
			}
		}

		Mat border_mask;
		Mat temp1, temp2;
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(mask, temp1, kernel);
		erode(mask, temp2, kernel);
		border_mask = temp1 - temp2;

		Mat border_mask_2 = border_mask.clone();
		dilate(border_mask_2, border_mask_2, kernel);

		Mat borders = Mat(color_correction.rows, color_correction.cols, CV_8UC3, Scalar(0, 0, 0));

		for(int i=0; i<color_correction.rows; i++) {
			for(int j=0; j<color_correction.cols; j++) {
				if(border_mask_2.at<uchar>(i, j) == 255) {
					borders.at<Vec3b>(i, j)[0] = color_correction.at<Vec3b>(i, j)[0];
					borders.at<Vec3b>(i, j)[1] = color_correction.at<Vec3b>(i, j)[1];
					borders.at<Vec3b>(i, j)[2] = color_correction.at<Vec3b>(i, j)[2];
				}
			}
		}

		Mat color_correction_smooth = color_correction.clone();
		GaussianBlur(borders, borders, Size(3, 3), 0, 0);

		for(int i=0; i<color_correction.rows; i++) {
			for(int j=0; j<color_correction.cols; j++) {
				if(border_mask.at<uchar>(i ,j) == 255) {
					color_correction_smooth.at<Vec3b>(i ,j)[0] = borders.at<Vec3b>(i, j)[0];
					color_correction_smooth.at<Vec3b>(i ,j)[1] = borders.at<Vec3b>(i, j)[1];
					color_correction_smooth.at<Vec3b>(i ,j)[2] = borders.at<Vec3b>(i, j)[2];
				}
			}
		}

		namedWindow("intensity_correction", WINDOW_NORMAL);
		namedWindow("umbra", WINDOW_NORMAL);
		namedWindow("penumbra", WINDOW_NORMAL);
		namedWindow("sunlight", WINDOW_NORMAL);
		namedWindow("color_correction", WINDOW_NORMAL);
		imshow("intensity_correction", intensity_correction);
		imshow("umbra", umbra);
		imshow("penumbra", penumbra);
		imshow("sunlight", sunlight);
		imshow("color_correction", color_correction);


		return color_correction_smooth;
	}
	
};

int main() {

	Mat img = imread("shadow_2.jpg", 1);
	Mat ycrcb, result, color_correction_result;
	Mat channels[3];
	Mat y_ch, y_ch_2;

	ShadowDetection detector(img);
	ycrcb = detector.convert_to_ycrcb();
	result = detector.detect();

	shadow_removal remover(img, result);
	color_correction_result=remover.remove();

	split(ycrcb, channels);
	y_ch = channels[0];

	equalizeHist(y_ch, y_ch_2);

	namedWindow("Input", WINDOW_NORMAL);
	namedWindow("YCrCb", WINDOW_NORMAL);
	namedWindow("Y", WINDOW_NORMAL);
	namedWindow("Y_2", WINDOW_NORMAL);
	namedWindow("Result", WINDOW_NORMAL);
	namedWindow("color_correction_result", WINDOW_NORMAL);

	imshow("Input", img);
	imshow("YCrCb", ycrcb);
	imshow("Y", y_ch);
	imshow("Y_2", y_ch_2);
	imshow("Result", result);
	imshow("color_correction_result", color_correction_result);

	waitKey(0);

	return 0;
}

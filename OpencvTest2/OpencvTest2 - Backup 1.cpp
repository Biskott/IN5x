// OpencvTest2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

// Header
Mat takePicture(int cameraId);
void takeTrainingPicture(int cameraId);
void training();
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened);
void setInDataBase(int value, Mat numberMat, vector<int> &classificationIntsTab, Mat &trainingImagesAsFlattened);
int initKNN();
int applyKNN(Mat pictureToCompare);
Mat getPolygon(vector<vector<Point> > contours, int areaIndex, Mat img_src, bool numberIsStraight);
Mat setSearchingArea(Mat img_src);
int getNumberInPicture(Mat pictureToCompare);
int getMilliCount();
int getMilliSpan(int nTimeStart);
void pictureToPolygons(Mat img_src, Mat &leftPicture, Mat &rightPicture);
Mat getStraightPolygon(Mat img_src, Rect pictureBoundRect, vector<Point> pictureContour);
float angleConversion(float angle);

vector<Point> getContour(Mat pictureToCompare);
vector<int> applyShapeRecognition(Mat pictureToCompare);
vector<int> KNNRange(Mat pictureToCompare);
void echanger(Point2d tableau[], int a, int b);
void quickSort(Point2d tableau[], int debut, int fin);
int averageNumberFound(vector<int> range1, vector<int> range2);
vector<int> rangePerimeter(Mat pictureToCompare);
vector<int> findNearestPerimeters(vector<Point2d> perimeters, double actualPerimeter);

Mat getPolygonWithHierarchy(vector<vector<Point> > contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src);

// Constants to resize pictures
const int RESIZED_IMAGE_WIDTH = 40;
const int RESIZED_IMAGE_HEIGHT = 60;

// Constant to set if the program made a new training 
const bool ASK_FOR_TRAINING_DEFAULT = false;

// Constant path
const string XML_PATH = "xml";
const string TRAINING_PICTURES_PATH = "new";

// Constant to filter size of polygons' deteted
int TOLERANCE_VALUE = 115;

// Search area in pourcentage of picture's size
int SEARCH_AREA_LEFT = 30;
int SEARCH_AREA_RIGHT = 60;
int SEARCH_AREA_TOP = 30;
int SEARCH_AREA_BOTTOM = 65;

// Local variable for training's matrix
Mat TrainingImagesAsFlattenedFloats;
Mat ClassificationInts;

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	int startTime;
	bool askForTraining = ASK_FOR_TRAINING_DEFAULT;

	// Set if training is required
	if (argc > 1) {
		if ((string)argv[1] == "true")
			askForTraining = true;
	}

	// Launch the training function
	if (askForTraining) {
		startTime = getMilliCount();
		cout << "--- Training started ---" << endl;
		training();
		cout << "Time takes to realize the training : " << getMilliSpan(startTime) << " milliseconds." << endl << endl;
	}

	// Loading xml
	startTime = getMilliCount();
	initKNN();
	cout << "Time token to load xml database :" << getMilliSpan(startTime) << endl;

	// Main loop
	int counter = 0;
	while (true) {
		Mat image;
		char key = 'q';
		while (key != ' ') {
			image = setSearchingArea(takePicture(1));
			imshow("Image prise", image);
			key = waitKey(10);
		}
		key = 'q';

		// Find Number's function with KNN method
		startTime = getMilliCount();
		int valueKNN = getNumberInPicture(image);
		cout << "Time token to find the good number with KNN method : " << getMilliSpan(startTime) << " milliseconds." << endl;

		// Find range of numbers with KNN method
		startTime = getMilliCount();
		Mat leftPicture, rightPicture;
		vector<int> knnRangeValuesLeft, knnRangeValuesRight;
		pictureToPolygons(image, leftPicture, rightPicture);
		knnRangeValuesRight = KNNRange(rightPicture);
		if(!leftPicture.empty())
			knnRangeValuesLeft = KNNRange(leftPicture);
		cout << "Time token to find the range of numbers with KNN methods : " << getMilliSpan(startTime) << " milliseconds." << endl;

		// Find range of numbers with shape method
		/*startTime = getMilliCount();
		vector<int> shapeRangeValues = applyShapeRecognition(image);
		cout << "Time token to find the range of numbers with shape method : " << getMilliSpan(startTime) << " milliseconds." << endl;

		// Find range of numbers with perimeter method
		startTime = getMilliCount();
		vector<int> perimeterRangeValues = rangePerimeter(image);
		cout << "Time token to find the range of numbers with perimeter method : " << getMilliSpan(startTime) << " milliseconds." << endl;

		// Display
		if (valueKNN == -1)
			cout << "No character detected with KNN methods ..." << endl;
		else
			cout << "Number read with KNN methods : " << valueKNN << endl;*/

		//cout << "Numbers found with shape method : " << shapeRangeValues[0] << endl;

		cout << "Range of values' found with KNN method : ";
		if (!knnRangeValuesLeft.empty()) {
			for (int i = 0; i < knnRangeValuesLeft.size(); ++i)
				cout << knnRangeValuesLeft[i] << " ";
			cout << "   ";
		}
		for (int i = 0; i < knnRangeValuesRight.size(); ++i)
			cout << knnRangeValuesRight[i] << " ";
		cout << endl;

		/*cout << "Range of values' found with shape method : ";
		for (int i = 0; i < shapeRangeValues.size(); ++i)
			cout << shapeRangeValues[i] << " ";
		cout << endl;
		cout << "Average number (made with KNN and shape ranges) : " << averageNumberFound(knnRangeValues, shapeRangeValues) << endl;

		cout << "Range of values' found with perimeter method : ";
		for (int i = 0; i < perimeterRangeValues.size(); ++i)
			cout << perimeterRangeValues[i] << " ";
		cout << endl;
		cout << "Average number (made with KNN and perimeter ranges) : " << averageNumberFound(knnRangeValues, perimeterRangeValues) << endl;*/

		key = waitKey();
		if (key == 27) {
			break;
		}
		else if (key == 'f') {
			cout << "Number : ";
			string value;
			cin >> value;
			cout << "Number 2 : ";
			string value2;
			cin >> value2;
			imwrite(TRAINING_PICTURES_PATH + "\\" + value + "_picture" + value2 + ".png", image);
			training();
			initKNN();
		}
		cout << endl;
	}
    return 0;
}
/*
 * Function to take a picture
 */
Mat takePicture(int cameraId) {

	VideoCapture capture(cameraId);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(CV_CAP_PROP_GAIN, 0);

	if (!capture.isOpened()) {
		exit(1);
	}
	Mat picture;
	capture >> picture;
	if (picture.empty()) {
		exit(2);
	}
	return picture;
}

void takeTrainingPicture(int cameraId) {

	char key = 'q';
	Mat image;

	while (key != 27) {
		image = setSearchingArea(takePicture(cameraId));
		imshow("Image prise", image);
		key = waitKey(10);

		if (key == 'w') {
			cout << "Number : ";
			string value;
			cin >> value;
			cout << "Number 2 : ";
			string value2;
			cin >> value2;
			imwrite(TRAINING_PICTURES_PATH + "\\" + value + "_picture" + value2 + ".png", image);
			key = 'q';
		}
	}
}

/*
* Training function for the KNN algorithm
*/
void training() {

	// Variable to stock training's datas 
	Mat trainingImagesAsFlattened;
	vector<int> classificationIntsTab;

	// List all the training pictures in the directory
	vector<string> trainingPicturesNames = Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);

	// Parse all pictures
	for (string s : trainingPicturesNames) {
		
		// Detect the number thanks to the file name
		int index = s.find_first_of('_');
		string number = s.substr(0, index);
		cout << "Training number read : " << number << endl;

		// Apply polygon detection
		Mat leftNumber, rightNumber;
		pictureToPolygons(imread(TRAINING_PICTURES_PATH + "\\" + s), leftNumber, rightNumber);

		// Set values in training data's variable
		if (number.length() == 1) {
			if (!rightNumber.empty()) {
				setInDataBase(number[0], rightNumber, classificationIntsTab, trainingImagesAsFlattened);
			}
		}
		else if (number.length() == 2) {
			if (!leftNumber.empty())
			{
				setInDataBase(number[0], leftNumber, classificationIntsTab, trainingImagesAsFlattened);
			}
			if (!rightNumber.empty())
			{
				setInDataBase(number[1], rightNumber, classificationIntsTab, trainingImagesAsFlattened);
			}
		}
	}
	// Save the two training xml files
	if (saveXmlTraining(classificationIntsTab, trainingImagesAsFlattened))
		cout << "Training xml files saved succesfully." << endl;
}

/*
* Function to save xml training files
*/
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened) {

	// Save xml number list
	FileStorage fsClassification(XML_PATH + "\\classification.xml", FileStorage::WRITE);
	if (fsClassification.isOpened()) {
		Mat classificationInts = Mat(classificationIntsTab);
		fsClassification << "classifications" << classificationInts;
		fsClassification.release();
	}
	else {
		return false;
	}
	// Save xml matrix's number list
	FileStorage fsTrainingImages(XML_PATH + "\\images.xml", FileStorage::WRITE);
	if (fsTrainingImages.isOpened()) {
		fsTrainingImages << "images" << trainingImagesAsFlattened;
		fsTrainingImages.release();
	}
	else {
		return false;
	}
	return true;
}

/*
* Function to add training picture's data in the different matrix
*/
void setInDataBase(int value, Mat numberMat, vector<int> &classificationIntsTab, Mat &trainingImagesAsFlattened)
{
	classificationIntsTab.push_back(value);

	Mat imageResized, imageFloat;
	resize(numberMat, imageResized, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	imageResized.convertTo(imageFloat, CV_32FC1);
	Mat imageFlattenedFloat = imageFloat.reshape(1, 1);
	trainingImagesAsFlattened.push_back(imageFlattenedFloat);
}

/*
* Function to init KNN algorithm (load xml training files)
*/
int initKNN() {

	// File which contain the classification of numbers
	FileStorage fsClassifications(XML_PATH + "\\classification.xml", FileStorage::READ);

	if (!fsClassifications.isOpened()) {
		cout << "Error, unable to open training classifications file..." << endl;
		return -1;
	}
	fsClassifications["classifications"] >> ClassificationInts;
	fsClassifications.release();

	// File which contain the picture's matrix of numbers
	FileStorage fsTrainingImages(XML_PATH + "\\images.xml", FileStorage::READ);

	if (fsTrainingImages.isOpened() == false) {
		cout << "Error, unable to open training images file..." << endl;
		return -2;
	}
	fsTrainingImages["images"] >> TrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();
}

/*
* Function to search which is the number (one digit) of the picture
*/
int applyKNN(Mat pictureToCompare) {

	CvMat training = TrainingImagesAsFlattenedFloats;											// NEW opencv 2.4
	CvMat classification = ClassificationInts;													// NEW opencv 2.4

	//Ptr<ml::KNearest> kNearest(ml::KNearest::create());										// OLD opencv 3.1
	//kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);		// OLD opencv 3.1
	int K = 1;																					// NEW opencv 2.4
	CvKNearest knn(&training, &classification, 0, false, K);									// NEW opencv 2.4
	CvMat* nearests = cvCreateMat(1, K, CV_32FC1);												// NEW opencv 2.4

	if(pictureToCompare.empty())
		return -3;

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat currentChar(0, 0, CV_32F);

	//kNearest->findNearest(pictureFloat, 1, currentChar);										// OLD opencv 3.1
	CvMat test = pictureFloat;																	// NEW opencv 2.4
	float fitCurrentChar = knn.find_nearest(&test, K, 0, 0, nearests, 0);						// NEW opencv 2.4

	//float fitCurrentChar = (float)currentChar.at<float>(0, 0);								// OLD opencv 3.1

	return int(fitCurrentChar) - 48;	// Because of ASCII
}

/*
* Function which return the polygon correspondant to the contour passed in parameters
* It also rotate polygon to have a straight number if bool is false
*/
Mat getPolygon(vector<vector<Point> > contours, int areaIndex, Mat img_src, bool numberIsStraight) {

	// Set the area of the number (with a straight rectangle)
	vector<vector<Point> > contours_poly(contours.size());
	approxPolyDP(Mat(contours[areaIndex]), contours_poly[areaIndex], 3, true);
	Rect boundRect = boundingRect(Mat(contours_poly[areaIndex]));

	// Draw this area in a empty matrix (with picture source's size)
	Scalar polygonColor = Scalar(255, 255, 255); // White contour
	img_src = Mat::zeros(img_src.size(), CV_8UC3);
	drawContours(img_src, contours_poly, areaIndex, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());

	// Crop picture 
	IplImage *frame = new IplImage(img_src);
	cvSetImageROI(frame, cvRect(boundRect.tl().x, boundRect.tl().y, boundRect.br().x - boundRect.tl().x, boundRect.br().y - boundRect.tl().y));
	Mat mat_res = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);

	// Put the number straight if it is not
	if (numberIsStraight) {
		return mat_res;
	}
	else {
		// Get straight polygon (a biggest picture to not lose parts of original picture)
		Mat straightPicture = getStraightPolygon(mat_res, boundRect, contours[areaIndex]);

		// Find contour with the new picture
		cvtColor(straightPicture, straightPicture, CV_BGR2GRAY);
		vector<vector<Point> > contours_straight;
		vector<Vec4i> hierarchy_straight;
		findContours(straightPicture, contours_straight, hierarchy_straight, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Get the index of the good contour
		int largestArea = 0, largestAreaIndex = 0;
		for (int i = 0; i < contours_straight.size(); i++)
		{
			double a = contourArea(contours_straight[i], false);
			if (a > largestArea) {
				largestArea = a;
				largestAreaIndex = i;
			}
		}
		// Call to getPolygon again to have a straight number polygon
		return getPolygon(contours_straight, largestAreaIndex, straightPicture, true);
	}
}

/*
* Function to set searching area
*/
Mat setSearchingArea(Mat img_src) {

	// Conversion in IplImage in order to use ROI
	IplImage *frame = new IplImage(img_src);

	// Calcul dimensions thanks to the percentage parameters
	int x = img_src.size().width*SEARCH_AREA_LEFT / 100;
	int y = img_src.size().height*SEARCH_AREA_TOP / 100;
	int width = img_src.size().width*SEARCH_AREA_RIGHT / 100 - x;
	int height = img_src.size().height*SEARCH_AREA_BOTTOM / 100 - y;

	// Set the new searching area
	cvSetImageROI(frame, cvRect(x, y, width, height));
	img_src = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);

	return img_src;
}

/*
* Function to find a number (with one or two digits) in a picture
*/
int getNumberInPicture(Mat pictureToCompare) {

	Mat leftPicture, rightPicture;

	pictureToPolygons(pictureToCompare, leftPicture, rightPicture);
	int valueLeft = applyKNN(leftPicture), valueRight = applyKNN(rightPicture);

	if (valueLeft < 0 && valueRight < 0) {
		return -1;
	}
	else {
		string result = "";
		if (valueLeft >= 0) {
			result += (valueLeft + 48);
		}
		if (valueRight >= 0) {
			result += (valueRight+48);
		}
		return stoi(result);
	}
}

/*
* Function get the actual time
*/
int getMilliCount() {
	timeb tb;
	ftime(&tb);
	int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
	return nCount;
}

/*
* Function to caculate the duration between now and an other time in parameter
*/
int getMilliSpan(int nTimeStart) {
	int nSpan = getMilliCount() - nTimeStart;
	if (nSpan < 0)
		nSpan += 0x100000 * 1000;
	return nSpan;
}

void pictureToPolygons(Mat img_src, Mat &leftPicture, Mat &rightPicture) {

	Mat img_temp;

	// Picture manipulation
	cvtColor(img_src, img_temp, CV_BGR2GRAY);
	blur(img_temp, img_temp, Size(3, 3));
	threshold(img_temp, img_temp, 127, 255, THRESH_BINARY);

	// Search for the different polygons in picture
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/*for (int i = 0; i < hierarchy.size(); ++i) {
		if (hierarchy[i][3] != -1) {
			cout << i << " : " << hierarchy[i] << endl;
		}
	}*/

	// Select the two biggest area
	int largestArea = 0, largestAreaIndex = 0, secondLargestArea = 0, secondLargestAreaIndex = 0;
	Rect boundRect1, boundRect2;
	for (int i = 0; i< contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largestArea) {
			Rect temp = boundingRect(contours[i]);
			if (temp.br().y - temp.tl().y >= TOLERANCE_VALUE) {
				// Move largest to secondLargest
				secondLargestArea = largestArea;
				secondLargestAreaIndex = largestAreaIndex;
				boundRect2 = boundRect1;

				// a = largest
				largestArea = a;
				largestAreaIndex = i;
				boundRect1 = boundingRect(contours[i]);
			}
		}
		else if (a > secondLargestArea) {
			Rect temp = boundingRect(contours[i]);
			if (temp.br().y - temp.tl().y >= TOLERANCE_VALUE) {
				// a = secondLargest
				secondLargestArea = a;
				secondLargestAreaIndex = i;
				boundRect2 = boundingRect(contours[i]);
			}
		}
	}
	// Set which polygon is right and which is left
	int leftAreaIndex, rightAreaIndex;
	int leftArea, rightArea;
	if (boundRect1.tl().x < boundRect2.tl().x) {
		leftAreaIndex = largestAreaIndex;
		leftArea = largestArea;
		rightAreaIndex = secondLargestAreaIndex;
		rightArea = secondLargestArea;
	}
	else {
		rightAreaIndex = largestAreaIndex;
		rightArea = largestArea;
		leftAreaIndex = secondLargestAreaIndex;
		leftArea = secondLargestArea;
	}

	// Set left picture
	if (leftArea == 0) {
		leftPicture = NULL;
	}
	else {
		//leftPicture = getPolygon(contours, leftAreaIndex, img_temp, false);
		leftPicture = getPolygonWithHierarchy(contours, leftAreaIndex, hierarchy, img_temp);

	}
	// Set right picture
	if (rightArea == 0) {
		rightPicture = NULL;
	}
	else {
		//rightPicture = getPolygon(contours, rightAreaIndex, img_temp, false);
		rightPicture = getPolygonWithHierarchy(contours, rightAreaIndex, hierarchy, img_temp);
	}
}

Mat getStraightPolygon(Mat img_src, Rect pictureBoundRect, vector<Point> pictureContour) {

	int marge = 30;
	RotatedRect rotRect = minAreaRect(pictureContour);
	Point2f centerOfPicture = Point2f(pictureBoundRect.width / 2.0F, pictureBoundRect.height / 2.0F);
	Mat rotMat = getRotationMatrix2D(centerOfPicture, angleConversion(rotRect.angle), 1);

	Mat img_res(img_src.rows+2*marge, img_src.cols+2*marge, img_src.type(), Scalar::all(0));
	img_src.copyTo(img_res(Rect(marge,marge, img_src.cols, img_src.rows)));

	warpAffine(img_res, img_res, rotMat, img_res.size() , INTER_CUBIC);

	return img_res;
}

float angleConversion(float angle) {

	if (angle < -45)
		return 90+angle;
	else
		return angle;
}

vector<int> applyShapeRecognition(Mat pictureToCompare) {

	vector<int> ints;
	vector<vector<Point>> points;
	vector<Point2d> correspondance;

	vector<string> trainingPicturesNames = Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);

	for (string s : trainingPicturesNames) {

		int index = s.find_first_of('_');
		string number = s.substr(0, index);

		Mat img = imread(TRAINING_PICTURES_PATH + "\\" + s);
		Mat L, R;
		pictureToPolygons(img, L, R);

		if (number.length() == 1) {
			if (!R.empty()) {
				ints.push_back(number[0]);
				points.push_back(getContour(R));
			}
		}
		else if (number.length() == 2) {
			if (!L.empty())
			{
				ints.push_back(number[0]);
				points.push_back(getContour(L));
			}
			if (!R.empty())
			{
				ints.push_back(number[1]);
				points.push_back(getContour(R));
			}
		}
	}
	//int bestMatch = 0;
	//double bestDis = FLT_MAX;
	Mat pict, temp;
	pictureToPolygons(pictureToCompare, pict, temp);
	pictureToCompare = temp;

	for (int i = 0; i < points.size(); i++) {

		double a = matchShapes(points[i], getContour(pictureToCompare), CV_CONTOURS_MATCH_I2, 0);
		correspondance.push_back(Point2d(ints[i]-48, a));
		/*if (a < bestDis) {
			bestDis = a;
			bestMatch = i;
		}*/
	}
	//cout << "Numbers found with shape method : " << ints[bestMatch]-48 << endl;

	imshow("source", Mat::zeros(Size(1, 1), pictureToCompare.type()));
	imshow("source", pictureToCompare);	

	quickSort(correspondance.data(), 0, correspondance.size() - 1);
	vector<int> res = vector<int>();
	for (int i = 0; i < 10; ++i) {
		res.push_back(correspondance[i].x);
		//cout << correspondance[i].x << " " << correspondance[i].y << endl;
	}
	/*vector<int> res = vector<int>();
	for (int i = 0; i < correspondance.size(); ++i) {
		if (correspondance[i].y < 1)
			res.push_back(correspondance[i].x);
	}*/
	return res;
}

vector<Point> getContour(Mat pictureToCompare) {
	int marge = 10;
	Mat img_res(pictureToCompare.rows + 2 * marge, pictureToCompare.cols + 2 * marge, pictureToCompare.type(), Scalar::all(0));
	pictureToCompare.copyTo(img_res(Rect(marge, marge, pictureToCompare.cols, pictureToCompare.rows)));

	//cvtColor(img_res, img_res, CV_BGR2GRAY);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_res, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Get the index of the good contour
	int largestArea = 0, largestAreaIndex = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largestArea) {
			largestArea = a;
			largestAreaIndex = i;
		}
	}
	return contours[largestAreaIndex];
}

vector<int> KNNRange(Mat pictureToCompare) {

	CvMat training = TrainingImagesAsFlattenedFloats;											// NEW opencv 2.4
	CvMat classification = ClassificationInts;													// NEW opencv 2.4

	//Ptr<ml::KNearest> kNearest(ml::KNearest::create());										// OLD opencv 3.1
	//kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);		// OLD opencv 3.1
	int K = 10;																					// NEW opencv 2.4
	CvKNearest knn(&training, &classification, 0, false, K);									// NEW opencv 2.4
	CvMat* nearests = cvCreateMat(1, K, CV_32FC1);												// NEW opencv 2.4

	if (pictureToCompare.empty())
		return vector<int>();

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat currentChar(0, 0, CV_32F);

	//kNearest->findNearest(pictureFloat, 1, currentChar);										// OLD opencv 3.1
	CvMat test = pictureFloat;																	// NEW opencv 2.4
	float fitCurrentChar = knn.find_nearest(&test, K, 0, 0, nearests, 0);						// NEW opencv 2.4

	vector<int> range;
	for (int i = 0; i < K; ++i) {
		range.push_back(nearests->data.fl[i] - 48);
	}

	//float fitCurrentChar = (float)currentChar.at<float>(0, 0);								// OLD opencv 3.1

	//return int(fitCurrentChar) - 48;	// Because of ASCII
	return range;
}

void echanger(Point2d tableau[], int a, int b){
	Point2d temp = tableau[a];
	tableau[a] = tableau[b];
	tableau[b] = temp;
}

void quickSort(Point2d tableau[], int debut, int fin){
	int gauche = debut - 1;
	int droite = fin + 1;
	const double pivot = tableau[debut].y;

	if (debut >= fin)
		return;

	while (1)
	{
		do droite--; while (tableau[droite].y > pivot);
		do gauche++; while (tableau[gauche].y < pivot);

		if (gauche < droite)
			echanger(tableau, gauche, droite);
		else break;
	}
	quickSort(tableau, debut, droite);
	quickSort(tableau, droite + 1, fin);
}

int averageNumberFound(vector<int> range1, vector<int> range2) {

	if (range1.size() != range2.size()) {
		return -1;
	}
	else {
		int counter = 1;
		while (range1.begin() + counter < range1.end()) {
			for (int i = 0; i < counter; ++i) {
				if (find(range2.begin(), range2.begin() + counter, range1[i]) != range2.begin() + counter) {
					return range1[i];
				}
			}
			counter++;
		}
	}
	/*for (int i = 0; i < range1.size(); ++i) {
		if (find(range2.begin(), range2.end(), range1[i]) != range2.end())
			return range1[i];
	}*/

	return -2;
}

vector<int> rangePerimeter(Mat pictureToCompare) {

	vector<Point2d> perimeters;

	vector<string> trainingPicturesNames = Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);

	for (string s : trainingPicturesNames) {

		int index = s.find_first_of('_');
		string number = s.substr(0, index);

		Mat img = imread(TRAINING_PICTURES_PATH + "\\" + s);
		Mat L, R;
		pictureToPolygons(img, L, R);

		if (number.length() == 1) {
			if (!R.empty()) {
				Point2d peri;
				peri.x = number[0]-48;
				peri.y = arcLength(getContour(R), true);
				perimeters.push_back(peri);
			}
		}
		else if (number.length() == 2) {
			if (!L.empty())
			{
				Point2d peri;
				peri.x = number[0]-48;
				peri.y = arcLength(getContour(L), true);
				perimeters.push_back(peri);
			}
			if (!R.empty())
			{
				Point2d peri;
				peri.x = number[1]-48;
				peri.y = arcLength(getContour(R), true);
				perimeters.push_back(peri);
			}
		}
	}

	Mat pict, temp;
	pictureToPolygons(pictureToCompare, pict, temp);
	pictureToCompare = temp;

	return findNearestPerimeters(perimeters, arcLength(getContour(pictureToCompare), true));
}

vector<int> findNearestPerimeters(vector<Point2d> perimeters, double actualPerimeter) {

	for (int i = 0; i < perimeters.size(); ++i) {
		perimeters[i].y = abs(perimeters[i].y - actualPerimeter);
	}

	quickSort(perimeters.data(), 0, perimeters.size() - 1);

	vector<int> res;

	for (int i = 0; i < /*perimeters.size()*/ 10; ++i) {
		/*if (perimeters[i].y > 20)
			break;*/
		res.push_back(perimeters[i].x);
		cout << perimeters[i].x  << " " << perimeters[i].y << endl;
	}
	return res;
}

Mat getPolygonWithHierarchy(vector<vector<Point>> contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src) {

	// Set the area of the number (with a straight rectangle)
	vector<vector<Point>> contours_poly(contours.size());
	approxPolyDP(Mat(contours[areaIndex]), contours_poly[areaIndex], 3, true);
	Rect boundRect = boundingRect(Mat(contours_poly[areaIndex]));

	// Draw this area in a empty matrix (with picture source's size)
	Scalar polygonColor = Scalar(255, 255, 255); // White contour
	img_src = Mat::zeros(img_src.size(), CV_8UC3);
	drawContours(img_src, contours_poly, areaIndex, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());

	for (int i = 0; i < hierarchy.size(); ++i) {
		if (hierarchy[i][3] == areaIndex) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			drawContours(img_src, contours_poly, i, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());
			//cout << "h " << i << " : " << hierarchy[i][3] << endl;
		}
	}

	// Crop picture 
	IplImage *frame = new IplImage(img_src);
	cvSetImageROI(frame, cvRect(boundRect.tl().x, boundRect.tl().y, boundRect.br().x - boundRect.tl().x, boundRect.br().y - boundRect.tl().y));
	Mat mat_res = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);

	// Get straight polygon (a biggest picture to not lose parts of original picture)
	Mat straightPicture = getStraightPolygon(mat_res, boundRect, contours[areaIndex]);

	// Find contour with the new picture
	vector<vector<Point>> contours_straight;
	vector<Vec4i> hierarchy_straight;
	cvtColor(straightPicture, straightPicture, CV_BGR2GRAY);
	findContours(straightPicture.clone(), contours_straight, hierarchy_straight, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Get the index of the good contour
	int largestArea = 0, largestAreaIndex = 0;
	for (int i = 0; i < contours_straight.size(); i++){
		double a = contourArea(contours_straight[i], false);
		if (a > largestArea) {
			largestArea = a;
			largestAreaIndex = i;
		}
	}
	vector<vector<Point> > contours_poly_straight(contours_straight.size());
	approxPolyDP(Mat(contours_straight[largestAreaIndex]), contours_poly_straight[largestAreaIndex], 3, true);
	Rect boundRectStraight = boundingRect(Mat(contours_poly_straight[largestAreaIndex]));

	// Crop picture
	IplImage *frameStraight = new IplImage(straightPicture);
	cvSetImageROI(frameStraight, cvRect(boundRectStraight.tl().x, boundRectStraight.tl().y, boundRectStraight.br().x - boundRectStraight.tl().x, boundRectStraight.br().y - boundRectStraight.tl().y));
	Mat res = (cvarrToMat(frameStraight).clone());
	cvResetImageROI(frameStraight);

	return res;
}
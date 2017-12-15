// ------------------- INCLUDE ----------------------

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


// ------------------- HEADER ----------------------

struct Picture{
	Mat image;
	int insideContourNumber;
};


// ------------------- HEADER ----------------------

// Take pictures
Mat takePicture(int cameraId);
void takeTrainingPicture(int cameraId);

// Training
void training();
bool saveXmlContourNumbers(map<int, set<int>> insideContourTable);
bool saveXmlPerimeter(vector<Point2d> perimeterDatas);
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened);
void setInDataBase(int value, Mat numberMat, vector<int> &classificationIntsTab, Mat &trainingImagesAsFlattened);

// Load training
int loadInsideContourValues();
int loadPerimeterValues();
int initKNN();

// Character recognition
int applyKNN(Mat pictureToCompare);
vector<int> KNNRange(Mat pictureToCompare);
vector<int> getAuthorizedNumbers(int insideContourNumber);
int averageNumberFound(vector<int> knnRange, vector<int> perimeterRange, vector <int> authorizedNumber);
bool getPerimeterRange(Mat pictureToCompare, vector<int> &perimeterRange);

// Picture modification
Mat setSearchingArea(Mat img_src);
void pictureToPolygons(Mat img_src, Picture &leftPicture, Picture &rightPicture, int thresholdValue);
Picture getPolygon(vector<vector<Point>> contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src);
Mat getStraightPolygon(Mat img_src, Rect pictureBoundRect, vector<Point> pictureContour);
vector<Point> getLargestContour(Mat pictureToCompare);
void cropPicture(Mat &pictureToCrop, Rect newArea);
Mat inverseColor(Mat picture);

// Main function
int getNumberInPicture(Mat pictureToCompare);

// Time elapsed
int getMilliCount();
int getMilliSpan(int nTimeStart);

// Angle conversion
float angleConversion(float angle);

// Vector contain
bool vectorContains(vector<int> tab, int value);

// Sort array
void swap(Point2d tableau[], int a, int b);
void sortTabByY(Point2d tableau[], int debut, int fin);

void CallBackFunc(int event, int x, int y, int flags, void *userdata);
Mat img; Point p1, p2; bool initPointSet;
Mat setSearchingArea2(Mat img_src, Point a, Point b);


// ------------------- CONSTANTS FOR THE PROGRAM ----------------------

// Constants to resize pictures
const int RESIZED_IMAGE_WIDTH = 40;
const int RESIZED_IMAGE_HEIGHT = 60;

// Constant threshold for picture manipulation
const int THRESHOLD_VALUE = 127;

// Constant to set if the program made a new training 
const bool ASK_FOR_TRAINING_DEFAULT = false;

// Constant path
const string XML_PATH = "xml";
const string TRAINING_PICTURES_PATH = "TrainingPictures";

// Constant filename
const string XML_PERIMETER_FILE_NAME = "perimeters.xml";
const string XML_CONTOUR_TABLE_FILE_NAME = "insideContourTable.xml";
const string XML_KNN_IMAGES_FILE_NAME = "images.xml";
const string XML_KNN_CLASSIFICATION_FILE_NAME = "classification.xml";

// Constant to filter size of polygons' deteted
const int TOLERANCE_VALUE = 115;

// Search area in pourcentage of picture's size
const int SEARCH_AREA_LEFT = 30;
const int SEARCH_AREA_RIGHT = 60;
const int SEARCH_AREA_TOP = 30;
const int SEARCH_AREA_BOTTOM = 65;

// K parameter for KNN range algorithm : nb of nearest matches in range
const int KNN_K_PARAMETER = 10;


// ------------------- LOCAL VARIABLES ----------------------

// Local variable for training's matrix
Mat TrainingImagesAsFlattenedFloats;
Mat ClassificationInts;
vector<Point2d> perimeterDatas;
map<int, vector<int>> insideContourTable;


// ------------------- FUNCTIONS ----------------------

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
		cout << "--- Training finished (" << getMilliSpan(startTime) << " ms) ---" << endl << endl;
	}

	// Loading KNN xml
	startTime = getMilliCount();
	if(initKNN()==0)
		cout << "KNN database's xml loaded (" << getMilliSpan(startTime) << " ms)" << endl;

	// Loading xml perimeter
	startTime = getMilliCount();
	if(loadPerimeterValues() == 0)
		cout << "Perimeters' xml loaded (" << getMilliSpan(startTime) << " ms)" << endl;

	// Loading xml inside contours
	startTime = getMilliCount();
	if(loadInsideContourValues()==0)
		cout << "Inside contours' xml loaded (" << getMilliSpan(startTime) << " ms)" << endl << endl;

	// Main loop
	cout << "--- Main loop has started ---" << endl;

	while (true) {
		cout << endl;
		Mat image;
		char key = 'q';
		while (key != ' ' && key != 27) {
			image = setSearchingArea(takePicture(1));
			imshow("Image prise", image);
			key = waitKey(10);
		}
		if (key == 27)
			break;

		Picture leftPicture, rightPicture; 
		pictureToPolygons(image, leftPicture, rightPicture, THRESHOLD_VALUE);


		// *********************** KNN methods **************************
		startTime = getMilliCount();
		int valueKNNLeft = applyKNN(leftPicture.image);
		if (valueKNNLeft >= 0)
			cout << "Left number read with KNN methods : " << valueKNNLeft << " (" << getMilliSpan(startTime) << " ms)" << endl;

		startTime = getMilliCount();
		int valueKNNRight = applyKNN(rightPicture.image);
		if(valueKNNRight >= 0)
			cout << "Right number read with KNN methods : " << valueKNNRight << " (" << getMilliSpan(startTime) << " ms)" << endl;


		// *********************** KNN methods with range **************************
		vector<int> knnRangeValuesLeft, knnRangeValuesRight;
		if (!leftPicture.image.empty()) {
			startTime = getMilliCount();
			knnRangeValuesLeft = KNNRange(leftPicture.image);
			cout << "Left number range with KNN method : ";
			for (int i = 0; i < knnRangeValuesLeft.size(); ++i)
				cout << knnRangeValuesLeft[i] << " ";
			cout << "(" << getMilliSpan(startTime) << " ms)" << endl;
			imshow("Left picture contour", leftPicture.image);
		}
		if (!rightPicture.image.empty()) {
			startTime = getMilliCount();
			knnRangeValuesRight = KNNRange(rightPicture.image);
			cout << "Right number range with KNN method : ";
			for (int i = 0; i < knnRangeValuesRight.size(); ++i)
				cout << knnRangeValuesRight[i] << " ";
			cout << "(" << getMilliSpan(startTime) << " ms)" << endl;
			imshow("Right picture contour", rightPicture.image);
		}


		// *********************** Perimeter methods with range **************************
		startTime = getMilliCount();
		vector<int> rangePerimeterListLeft, rangePerimeterListRight;
		if (!leftPicture.image.empty()) {
			getPerimeterRange(leftPicture.image, rangePerimeterListLeft);
			cout << "Left number range with perimeter method : ";
			for (int i = 0; i < rangePerimeterListLeft.size(); ++i)
				cout << rangePerimeterListLeft[i] << " ";
			cout << "(" << getMilliSpan(startTime) << " ms)" << endl;
		}
		if (!rightPicture.image.empty()) {
			getPerimeterRange(rightPicture.image, rangePerimeterListRight);
			cout << "Right number range with perimeter method : ";
			for (int i = 0; i < rangePerimeterListRight.size(); ++i)
				cout << rangePerimeterListRight[i] << " ";
			cout << "(" << getMilliSpan(startTime) << " ms)" << endl;
		}


		// *********************** Full function (KNN + perimeter + insideContour) **************************
		startTime = getMilliCount();
		int getNumberValue = getNumberInPicture(image);
		cout << "Number read with full function : " << getNumberValue << " (" << getMilliSpan(startTime) << " ms)" << endl;
	}

	destroyAllWindows();
	img = takePicture(1);
	namedWindow("Test cursor detection");
	setMouseCallback("Test cursor detection", CallBackFunc, NULL);
	imshow("Test cursor detection", img);
	initPointSet = false;
	while (waitKey(10) != ' ') {}
	destroyAllWindows();
	imshow("New picture area", setSearchingArea2(img, p1, p2));
	waitKey();

    return 0;
}


Mat setSearchingArea2(Mat img_src, Point a, Point b) {

	// Conversion in IplImage in order to use ROI
	IplImage *frame = new IplImage(img_src);

	// Set the new searching area
	cvSetImageROI(frame, cvRect(a.x, a.y, b.x - a.x, b.y - a.y));
	img_src = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);

	return img_src;
}

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {

	if (event == EVENT_LBUTTONDOWN) {
		p1 = Point(x, y);
		initPointSet = true;
	}
	if (event == EVENT_LBUTTONUP) {
		Mat new_img = img.clone();
		p2 = Point(x, y);
		rectangle(new_img, p1, p2, Scalar(0, 0, 255), 2, 8, 0);
		imshow("Test cursor detection", new_img);
		initPointSet = false;
	}
	if (event == EVENT_MOUSEMOVE && initPointSet) {
		Mat temp = img.clone();
		rectangle(temp, p1, Point(x, y), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Test cursor detection", temp);
	}
}

/*//
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
			imwrite(TRAINING_PICTURES_PATH + "\\" + value + "_picture" + ".png", image);
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
	vector<Point2d> perimeterDatas;
	map<int, set<int>> insideContoursTable;

	// List all the training pictures in the directory
	vector<string> trainingPicturesNames = Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);

	// Parse all pictures
	for (string s : trainingPicturesNames) {
		
		// Detect the number thanks to the file name
		int index = s.find_first_of('_');
		string number = s.substr(0, index);
		cout << "Training number read : " << number << endl;

		// Apply polygon detection
		Picture leftNumber, rightNumber; /*leftNumber.insideContourNumber = -1; rightNumber.insideContourNumber = -1;*/
		pictureToPolygons(imread(TRAINING_PICTURES_PATH + "\\" + s), leftNumber, rightNumber, THRESHOLD_VALUE);

		// Set values in training data's variable
		if (number.length() == 1) {
			if (!rightNumber.image.empty()) {
				setInDataBase(number[0]-48, rightNumber.image, classificationIntsTab, trainingImagesAsFlattened);
				perimeterDatas.push_back(Point2d(number[0]-48, arcLength(getLargestContour(rightNumber.image), true)));
				insideContoursTable[rightNumber.insideContourNumber].insert(number[0]-48);
			}
		}
		else if (number.length() == 2) {
			if (!leftNumber.image.empty())
			{
				setInDataBase(number[0]-48, leftNumber.image, classificationIntsTab, trainingImagesAsFlattened);
				perimeterDatas.push_back(Point2d(number[0]-48, arcLength(getLargestContour(leftNumber.image), true)));
			}
			if (!rightNumber.image.empty())
			{
				setInDataBase(number[1]-48, rightNumber.image, classificationIntsTab, trainingImagesAsFlattened);
				perimeterDatas.push_back(Point2d(number[1]-48, arcLength(getLargestContour(rightNumber.image), true)));
				insideContoursTable[rightNumber.insideContourNumber].insert(number[1]-48);
			}
		}
	}
	// Save the two training xml files
	if (saveXmlTraining(classificationIntsTab, trainingImagesAsFlattened))
		cout << "Training xml files for KNN saved succesfully." << endl;

	// Save the xml perimeters file
	if (saveXmlPerimeter(perimeterDatas)) {
		cout << "Training xml file for perimeter range saved succesfully" << endl;
	}
	// Save the xml inside contour numbers file
	if (saveXmlContourNumbers(insideContoursTable)) {
		cout << "Training xml file for contour's numbers saved succesfully" << endl;
	}
}

/*
* Function to save xml training files for inside contours' numbers
*/
bool saveXmlContourNumbers(map<int, set<int>> insideContourTable) {


	FileStorage fsContourNumberDatas(XML_PATH + "\\" + XML_CONTOUR_TABLE_FILE_NAME, FileStorage::WRITE);

	if (fsContourNumberDatas.isOpened()) {

		for (int i = 0; i < insideContourTable.size() ; ++i) {
			stringstream data, tag;
			tag << "contour" << i;
			set<int>::const_iterator sit(insideContourTable[i].begin()), send(insideContourTable[i].end());
			for (; sit != send;++sit) {
				data << *sit;
			}
			fsContourNumberDatas << tag.str() << data.str();
		}
		fsContourNumberDatas.release();
		return true;
	}
	else {
		return false;
	}
}

/*
* Function to save xml training files for perimeter range
*/
bool saveXmlPerimeter(vector<Point2d> perimeterDatas) {

	FileStorage fsPerimeterDatas(XML_PATH + "\\" + XML_PERIMETER_FILE_NAME, FileStorage::WRITE);

	if (fsPerimeterDatas.isOpened()) {
		for (int i = 0; i < perimeterDatas.size();++i) {
			stringstream data, tag;
			tag << "perimeter" << i;
			data << perimeterDatas[i].x << ":" << perimeterDatas[i].y;
			fsPerimeterDatas << tag.str() << data.str();
		}
		fsPerimeterDatas.release();
		return true;
	}
	else {
		return false;
	}
}

/*
* Function to save xml training files for KNN algorithm
*/
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened) {

	// Save xml number list
	FileStorage fsClassification(XML_PATH + "\\" + XML_KNN_CLASSIFICATION_FILE_NAME, FileStorage::WRITE);
	if (fsClassification.isOpened()) {
		Mat classificationInts = Mat(classificationIntsTab);
		fsClassification << "classifications" << classificationInts;
		fsClassification.release();
	}
	else {
		return false;
	}
	// Save xml matrix's number list
	FileStorage fsTrainingImages(XML_PATH + "\\" + XML_KNN_IMAGES_FILE_NAME, FileStorage::WRITE);
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
* Function to load inside contour values from xml file
*/
int loadInsideContourValues() {

	// File which contain the classification of numbers
	FileStorage fsContourNumberDatas(XML_PATH + "\\" + XML_CONTOUR_TABLE_FILE_NAME, FileStorage::READ);

	if (!fsContourNumberDatas.isOpened()) {
		cout << "Error, unable to open training inside contour file..." << endl;
		return -1;
	}
	insideContourTable = map<int, vector<int>>();

	int i = 0;
	string value = "Init";
	stringstream tag;
	while (!value.empty())
	{
		tag << "contour" << i;
		fsContourNumberDatas[tag.str()] >> value;

		insideContourTable[i] = vector<int>();
		for (char number : value) {
			insideContourTable[i].push_back(number-48);
		}

		tag = stringstream();
		++i;
	}
	fsContourNumberDatas.release();

	if (i == 1) {
		cout << "Error, no datas in inside contours file..." << endl;
		return -2;
	}
	return 0;
}

/*
* Function to load perimeter values from xml file
*/
int loadPerimeterValues() {

	// File which contain the classification of numbers
	FileStorage fsPerimeterDatas(XML_PATH + "\\" + XML_PERIMETER_FILE_NAME, FileStorage::READ);

	if (!fsPerimeterDatas.isOpened()) {
		cout << "Error, unable to open training perimeter file..." << endl;
		return -1;
	}
	perimeterDatas = vector<Point2d>();

	int i = 0;
	string value = "Init";
	stringstream tag;
	while (!value.empty())
	{
		tag << "perimeter" << i;
		fsPerimeterDatas[tag.str()] >> value;

		int Index = value.find(':');
		if(!value.empty())
			perimeterDatas.push_back(Point2d(stod(value.substr(0, Index)), stod(value.substr(Index+1, value.size()-Index-1))));

		tag = stringstream();
		++i;
	}
	fsPerimeterDatas.release();

	if (i == 1) {
		cout << "Error, no datas in perimeter file..." << endl;
		return -2;
	}
	return 0;
}

/*
* Function to init KNN algorithm (load xml training files)
*/
int initKNN() {

	// File which contain the classification of numbers
	FileStorage fsClassifications(XML_PATH + "\\" + XML_KNN_CLASSIFICATION_FILE_NAME, FileStorage::READ);

	if (!fsClassifications.isOpened()) {
		cout << "Error, unable to open training classifications file..." << endl;
		return -1;
	}
	fsClassifications["classifications"] >> ClassificationInts;
	fsClassifications.release();

	// File which contain the picture's matrix of numbers
	FileStorage fsTrainingImages(XML_PATH + "\\" + XML_KNN_IMAGES_FILE_NAME, FileStorage::READ);

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

	return int(fitCurrentChar);
}

/*
* Function to search the best match numbers (range) in the picture (one digit)
*/
vector<int> KNNRange(Mat pictureToCompare) {

	CvMat training = TrainingImagesAsFlattenedFloats;
	CvMat classification = ClassificationInts;
																				
	CvKNearest knn(&training, &classification, 0, false, KNN_K_PARAMETER);	
	CvMat* nearests = cvCreateMat(1, KNN_K_PARAMETER, CV_32FC1);

	if (pictureToCompare.empty())
		return vector<int>();

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat currentChar(0, 0, CV_32F);

	CvMat test = pictureFloat;
	float fitCurrentChar = knn.find_nearest(&test, KNN_K_PARAMETER, 0, 0, nearests, 0);

	vector<int> range;
	for (int i = 0; i < KNN_K_PARAMETER; ++i) {
		range.push_back(nearests->data.fl[i]);
	}

	return range;
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

void pictureToPolygons(Mat img_src, Picture &leftPicture, Picture &rightPicture, int thresholdValue) {

	Mat img_temp;

	// Picture manipulation
	cvtColor(img_src, img_temp, CV_BGR2GRAY);
	blur(img_temp, img_temp, Size(3, 3));
	threshold(img_temp, img_temp, thresholdValue, 255, THRESH_BINARY);

	// Search for the different polygons in picture
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

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
		leftPicture.image = NULL;
		leftPicture.insideContourNumber = 0;
	}
	else {
		leftPicture = getPolygon(contours, leftAreaIndex, hierarchy, img_src);
	}
	// Set right picture
	if (rightArea == 0) {
		rightPicture.image = NULL;
		rightPicture.insideContourNumber = 0;
	}
	else {
		rightPicture = getPolygon(contours, rightAreaIndex, hierarchy, img_src);
	}
}

/*
* Function which return the polygon correspondant to the contour passed in parameters
* It also rotate polygon to have a straight number if bool is false
*/
Picture getPolygon(vector<vector<Point>> contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src) {

	// Set the area of the number (with a straight rectangle)
	vector<vector<Point>> contours_poly(contours.size());
	approxPolyDP(Mat(contours[areaIndex]), contours_poly[areaIndex], 3, true);
	Rect boundRect = boundingRect(Mat(contours_poly[areaIndex]));

	// Draw this area in a empty matrix (with picture source's size)
	Scalar polygonColor = Scalar(255, 255, 255); // White contour

	// Crop picture source picture
	cropPicture(img_src, boundRect);

	// Get straight polygon (a biggest picture to not lose parts of original picture)
	Mat straightPicture = getStraightPolygon(img_src, boundRect, contours[areaIndex]);

	// Picture manipulation on new picture
	cvtColor(straightPicture, straightPicture, CV_BGR2GRAY);
	blur(straightPicture, straightPicture, Size(3, 3));
	threshold(straightPicture, straightPicture, 127, 255, THRESH_BINARY);

	// Find contour with the new picture
	vector<vector<Point>> contours_straight;
	vector<Vec4i> hierarchy_straight;
	findContours(straightPicture, contours_straight, hierarchy_straight, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Get the index the extern contour
	int largestArea = 0, largestAreaIndex = 0;
	for (int i = 0; i < contours_straight.size(); i++) {
		double a = contourArea(contours_straight[i], false);
		if (a > largestArea) {
			largestArea = a;
			largestAreaIndex = i;
		}
	}
	// Detect the area of the number in the picture
	vector<vector<Point> > contours_poly_straight(contours_straight.size());
	approxPolyDP(Mat(contours_straight[largestAreaIndex]), contours_poly_straight[largestAreaIndex], 3, true);
	Rect boundRectStraight = boundingRect(Mat(contours_poly_straight[largestAreaIndex]));

	// Draw extern contour in a empty matrix (with straight picture's size)
	Picture resPicture;
	resPicture.image = Mat::zeros(straightPicture.size(), CV_8UC3);
	drawContours(resPicture.image, contours_poly_straight, largestAreaIndex, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());

	// Draw intern contour
	int index = 0;
	for (int i = 0; i < hierarchy_straight.size(); ++i) {
		if (hierarchy_straight[i][3] == largestAreaIndex) {
			approxPolyDP(Mat(contours_straight[i]), contours_poly_straight[i], 3, true);
			drawContours(resPicture.image, contours_poly_straight, i, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());
			index++;
		}
	}
	resPicture.insideContourNumber = index;

	// Crop picture
	cropPicture(resPicture.image, boundRectStraight);

	return resPicture;
}

/*
* Function to find a number (with one or two digits) in a picture
*/
int getNumberInPicture(Mat pictureToCompare) {

	Picture leftPicture, rightPicture;

	pictureToPolygons(pictureToCompare, leftPicture, rightPicture, THRESHOLD_VALUE);
	int valueLeft, valueRight;

	vector<int> knnLeft = KNNRange(leftPicture.image), knnRight = KNNRange(rightPicture.image);

	vector<int> rangePerimeterListLeft, rangePerimeterListRight;
	getPerimeterRange(leftPicture.image, rangePerimeterListLeft);
	getPerimeterRange(rightPicture.image, rangePerimeterListRight);

	valueLeft = averageNumberFound(knnLeft, rangePerimeterListLeft, getAuthorizedNumbers(leftPicture.insideContourNumber));
	valueRight = averageNumberFound(knnRight, rangePerimeterListRight, getAuthorizedNumbers(rightPicture.insideContourNumber));

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

vector<Point> getLargestContour(Mat pictureToCompare) {
	int marge = 10;
	Mat img_res(pictureToCompare.rows + 2 * marge, pictureToCompare.cols + 2 * marge, pictureToCompare.type(), Scalar::all(0));
	pictureToCompare.copyTo(img_res(Rect(marge, marge, pictureToCompare.cols, pictureToCompare.rows)));

	if (img_res.channels()>1)
		cvtColor(img_res, img_res, CV_BGR2GRAY);
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

void swap(Point2d tableau[], int a, int b){
	Point2d temp = tableau[a];
	tableau[a] = tableau[b];
	tableau[b] = temp;
}

void sortTabByY(Point2d tableau[], int debut, int fin){
	int left = debut - 1;
	int right = fin + 1;
	const double pivot = tableau[debut].y;

	if (debut >= fin)
		return;

	while (true)
	{
		do right--; while (tableau[right].y > pivot);
		do left++; while (tableau[left].y < pivot);

		if (left < right)
			swap(tableau, left, right);
		else break;
	}
	sortTabByY(tableau, debut, right);
	sortTabByY(tableau, right + 1, fin);
}

int averageNumberFound(vector<int> knnRange, vector<int> perimeterRange, vector<int> authorizedNumber) {

	for (int knnValue : knnRange) {

		if (vectorContains(authorizedNumber, knnValue)) {

			for (int perimeterValue : perimeterRange) {
				if (knnValue == perimeterValue)
					return knnValue;
			}
		}
	}
	return -1;
}

bool getPerimeterRange(Mat pictureToCompare, vector<int> &perimeterRange) {

	if(!pictureToCompare.empty()){

		vector<Point2d> perimeterScore;

		double actualPerimeter = arcLength(getLargestContour(pictureToCompare), true);

		for (int i = 0; i < perimeterDatas.size(); ++i) {

			perimeterScore.push_back(Point2d(perimeterDatas[i].x, abs(perimeterDatas[i].y - actualPerimeter)));
		}

		perimeterRange = vector<int>();

		sortTabByY(perimeterScore.data(), 0, perimeterScore.size() - 1);

		for (int i = 0; i < perimeterScore.size(); ++i) {
			if (perimeterScore[i].y > 20)
				break;
			perimeterRange.push_back(perimeterScore[i].x);
		}
		return true;
	}
	return false;
}

void cropPicture(Mat &pictureToCrop, Rect newArea) {

	IplImage *frame = new IplImage(pictureToCrop);
	cvSetImageROI(frame, cvRect(newArea.tl().x, newArea.tl().y, newArea.br().x - newArea.tl().x, newArea.br().y - newArea.tl().y));
	pictureToCrop = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);
}

bool vectorContains(vector<int> tab, int value) {

	for (int tabValue : tab) {
		if (tabValue == value)
			return true;
	}
	return false;
}

vector<int> getAuthorizedNumbers(int insideContourNumber) {

	if (insideContourTable.find(insideContourNumber) != insideContourTable.end())
		return insideContourTable[insideContourNumber];
	return vector<int>();
}

/*
* Function to make a negative of a picture
*/
Mat inverseColor(Mat picture) {

	Mat new_img = Mat::zeros(picture.size(), picture.type());
	Mat sub_mat = Mat::ones(picture.size(), picture.type()) * 255;
	subtract(sub_mat, picture, new_img);

	return new_img;
}
#include "stdafx.h"
#include "CharacterRecognition.h"


// ------------------- GLOBAL VARIABLES ----------------------

Mat TrainingImagesAsFlattenedFloats;
Mat ClassificationInts;
vector<Point2d> perimeterDatas;
map<int, vector<int>> insideContourTable;


/*
* Function to search which is the number (one digit) of the picture
*/
int applyKNN(Mat pictureToCompare) {

	Ptr<ml::KNearest> kNearest(ml::KNearest::create());
	kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);

	if (pictureToCompare.empty())
		return -3;

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat currentChar(0, 0, CV_32F);

	kNearest->findNearest(pictureFloat, 1, currentChar);

	float fitCurrentChar = (float)currentChar.at<float>(0, 0);

	return int(fitCurrentChar);
}

/*
* Function to search the best match numbers (range) in the picture (one digit)
*/
vector<Point2i> KNNRange(Mat pictureToCompare) {

	Mat training = TrainingImagesAsFlattenedFloats;
	Mat classification = ClassificationInts;

	Ptr<ml::KNearest> kNearest(ml::KNearest::create());
	kNearest->setDefaultK(KNN_K_PARAMETER);
	kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);

	if (pictureToCompare.empty())
		return vector<Point2i>();
	
	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);
	
	Mat nearest(0, 0, CV_32F);
	Mat dist(0, 0, CV_32F);
	kNearest->findNearest(pictureFloat, KNN_K_PARAMETER, nearest, noArray(), dist);
	
	vector<Point2i> range;
	for (int i = 0; i < KNN_K_PARAMETER; ++i) {
		int percent = (MAX_KNN_VALUE - (float)dist.at<float>(0, i)) / MAX_KNN_VALUE * 100;
		range.push_back(Point2i((float)nearest.at<float>(0, i), percent));
	}
	return range;
}

/*
* Function to get the best number found
*/
Point2i averageNumberFound(vector<Point2i> knnRange, vector<int> authorizedNumber) {

	for (Point2i knnValue : knnRange) {

		if (vectorContains(authorizedNumber, knnValue.x)) {
			return knnValue;
		}
	}
	return Point2i(-1,0);
}

/*
* Function to get a list of authorized numbers thanks to the number of intern contours
*/
vector<int> getAuthorizedNumbers(int insideContourNumber) {

	if (insideContourTable.find(insideContourNumber) != insideContourTable.end())
		return insideContourTable[insideContourNumber];
	return vector<int>();
}

/*
* Function to find a number (with one or two digits) in a picture
*/
int getNumberInPicture(Mat pictureToCompare, float &percentage) {

	Picture leftPicture, rightPicture;
	
	pictureToPolygons(pictureToCompare, leftPicture, rightPicture, THRESHOLD_VALUE);
	Point2i valueLeft, valueRight;
	
	vector<Point2i> knnLeft = KNNRange(leftPicture.image), knnRight = KNNRange(rightPicture.image);

	valueLeft = averageNumberFound(knnLeft, getAuthorizedNumbers(leftPicture.insideContourNumber));
	valueRight = averageNumberFound(knnRight, getAuthorizedNumbers(rightPicture.insideContourNumber));

	if (valueLeft.x < 0 && valueRight.x < 0) {
		return -1;
	}
	else {
		percentage = -1;
		string result = "";
		if (valueLeft.x >= 0) {
			result += (valueLeft.x + 48);
			percentage = valueLeft.y;
			cout << "Left number percentage correspondance : " << valueLeft.y << "%" << endl;
		}
		if (valueRight.x >= 0) {
			result += (valueRight.x + 48);
			cout << "Right number percentage correspondance : " << valueRight.y << "%" << endl;
			if (percentage != -1)
				percentage = (percentage + float(valueRight.y)) / 2;
			else
				percentage = valueRight.y;
		}
		return stoi(result);
	}
}
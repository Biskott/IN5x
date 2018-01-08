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
Point2i applyKNN(Mat pictureToCompare, vector<int> authorizedNumber) {

	Mat training/* = TrainingImagesAsFlattenedFloats*/;
	Mat classification/* = ClassificationInts*/;

	for (int i = 0; i < ClassificationInts.rows; ++i) {
		if (vectorContains(authorizedNumber, ClassificationInts.at<int>(i, 0))) {
			training.push_back(TrainingImagesAsFlattenedFloats.row(i));
			classification.push_back(ClassificationInts.at<int>(i, 0));
		}
	}

	Ptr<ml::KNearest> kNearest(ml::KNearest::create());
	kNearest->setDefaultK(KNN_K_PARAMETER);
	kNearest->train(training, ml::ROW_SAMPLE, classification);

	if (pictureToCompare.empty())
		return Point2i(-1,-1);
	
	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);
	
	Mat nearest(0, 0, CV_32F);
	Mat dist(0, 0, CV_32F);
	kNearest->findNearest(pictureFloat, KNN_K_PARAMETER, nearest, noArray(), dist);

	int percent = (MAX_KNN_VALUE - (float)dist.at<float>(0, 0)) / MAX_KNN_VALUE * 100;

	return Point2i((float)nearest.at<float>(0, 0), percent);
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
	
	Point2i valueLeft = applyKNN(leftPicture.image, getAuthorizedNumbers(leftPicture.insideContourNumber)),
		valueRight = applyKNN(rightPicture.image, getAuthorizedNumbers(rightPicture.insideContourNumber));

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
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

	//CvMat training = TrainingImagesAsFlattenedFloats;											// NEW opencv 2.4
	//CvMat classification = ClassificationInts;													// NEW opencv 2.4

	Ptr<ml::KNearest> kNearest(ml::KNearest::create());										// OLD opencv 3.1
	kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);		// OLD opencv 3.1
	//int K = 1;																					// NEW opencv 2.4
	//CvKNearest knn(&training, &classification, 0, false, K);									// NEW opencv 2.4
	//CvMat* nearests = cvCreateMat(1, K, CV_32FC1);												// NEW opencv 2.4

	if (pictureToCompare.empty())
		return -3;

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat currentChar(0, 0, CV_32F);

	kNearest->findNearest(pictureFloat, 1, currentChar);										// OLD opencv 3.1
	//CvMat test = pictureFloat;																	// NEW opencv 2.4
	//float fitCurrentChar = knn.find_nearest(&test, K, 0, 0, nearests, 0);						// NEW opencv 2.4

	float fitCurrentChar = (float)currentChar.at<float>(0, 0);								// OLD opencv 3.1

	return int(fitCurrentChar);
}

/*
* Function to search the best match numbers (range) in the picture (one digit)
*/
vector<Point2i> KNNRange(Mat pictureToCompare) {

	CvMat training = TrainingImagesAsFlattenedFloats;
	CvMat classification = ClassificationInts;

	Ptr<ml::KNearest> kNearest(ml::KNearest::create());
	kNearest->train(TrainingImagesAsFlattenedFloats, ml::ROW_SAMPLE, ClassificationInts);

	//CvKNearest knn(&training, &classification, 0, false, KNN_K_PARAMETER);
	//CvMat* nearests = cvCreateMat(1, KNN_K_PARAMETER, CV_32FC1);

	if (pictureToCompare.empty())
		return vector<Point2i>();

	Mat pictureFloat;
	resize(pictureToCompare, pictureToCompare, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
	pictureToCompare.convertTo(pictureFloat, CV_32FC1);
	pictureFloat = pictureFloat.reshape(1, 1);

	Mat nearest(0, 0, CV_32F);
	Mat dist(0, 0, CV_32F);
	kNearest->findNearest(pictureFloat, KNN_K_PARAMETER, nearest, noArray(), dist);

	//CvMat test = pictureFloat;
	//CvMat *dist = cvCreateMat(1, KNN_K_PARAMETER, CV_32FC1);
	//float fitCurrentChar = knn.find_nearest(&test, KNN_K_PARAMETER, 0, 0, nearests, dist);

	vector<Point2i> range;
	for (int i = 0; i < KNN_K_PARAMETER; ++i) {
		int percent = (MAX_KNN_VALUE - (float)dist.at<float>(0, i)) / MAX_KNN_VALUE * 100;
		range.push_back(Point2i((float)nearest.at<float>(0, i), percent));
	}

	return range;
}

/*
* Function to get the best best number found
*/
Point2i averageNumberFound(vector<Point2i> knnRange, vector<int> perimeterRange, vector<int> authorizedNumber) {

	for (Point2i knnValue : knnRange) {

		if (vectorContains(authorizedNumber, knnValue.x)) {

			if (USE_PERIMETER) {

				for (int perimeterValue : perimeterRange) {
					if (knnValue.x == perimeterValue)
						return knnValue;
				}
			}
			else {
				return knnValue;
			}
		}
	}
	return Point2i(-1,0);
}

/*
* Function to get range perimeter
*/
bool getPerimeterRange(Mat pictureToCompare, vector<int> &perimeterRange) {

	if (!pictureToCompare.empty()) {

		vector<Point2d> perimeterScore;

		double actualPerimeter;
		if (PERIMETER_RESIZE) {
			vector<Point> resizePerimeter = getLargestContour(pictureToCompare);
			resizePoints(resizePerimeter, pictureToCompare.rows, pictureToCompare.cols, RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT);
			actualPerimeter = arcLength(resizePerimeter, true);
		}
		else {
			actualPerimeter = arcLength(getLargestContour(pictureToCompare), true);
		}

		for (int i = 0; i < perimeterDatas.size(); ++i) {

			perimeterScore.push_back(Point2d(perimeterDatas[i].x, abs(perimeterDatas[i].y - actualPerimeter)));
		}

		perimeterRange = vector<int>();

		sortTabByY(perimeterScore.data(), 0, perimeterScore.size() - 1);

		for (int i = 0; i < perimeterScore.size(); ++i) {
			if (perimeterScore[i].y > PERIMETER_TOLERANCE)
				break;
			perimeterRange.push_back(perimeterScore[i].x);
		}
		return true;
	}
	return false;
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
int getNumberInPicture(Mat pictureToCompare) {

	Picture leftPicture, rightPicture;

	pictureToPolygons(pictureToCompare, leftPicture, rightPicture, THRESHOLD_VALUE);
	Point2i valueLeft, valueRight;

	vector<Point2i> knnLeft = KNNRange(leftPicture.image), knnRight = KNNRange(rightPicture.image);

	vector<int> rangePerimeterListLeft, rangePerimeterListRight;

	if (USE_PERIMETER) {
		getPerimeterRange(leftPicture.image, rangePerimeterListLeft);
		getPerimeterRange(rightPicture.image, rangePerimeterListRight);
	}

	valueLeft = averageNumberFound(knnLeft, rangePerimeterListLeft, getAuthorizedNumbers(leftPicture.insideContourNumber));
	valueRight = averageNumberFound(knnRight, rangePerimeterListRight, getAuthorizedNumbers(rightPicture.insideContourNumber));

	if (valueLeft.x < 0 && valueRight.x < 0) {
		return -1;
	}
	else {
		string result = "";
		if (valueLeft.x >= 0) {
			result += (valueLeft.x + 48);
			cout << "Left number percentage correspondance : " << valueLeft.y << "%" << endl;
		}
		if (valueRight.x >= 0) {
			result += (valueRight.x + 48);
			cout << "Right number percentage correspondance : " << valueRight.y << "%" << endl;
		}
		return stoi(result);
	}
}
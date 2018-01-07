#include "stdafx.h"
#include "PictureModification.h"

/*
* Function to get the contour of the one or two polygons present in the picture
*/
void pictureToPolygons(Mat img_src, Picture &leftPicture, Picture &rightPicture, int thresholdValue) {

	Mat img_temp;

	// Picture manipulation
	cvtColor(img_src, img_temp, CV_BGR2GRAY);

	// Do we want black pixels on white background or the opposite ?
	if (INVERSE_PICTURE)
		img_temp = inverseColor(img_temp);
	Mat thresh;

	blur(img_temp, img_temp, Size(3, 3));
	// blur(img_temp, img_temp, Size(3, 3));

	threshold(img_temp, img_temp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	int dil_size = 2;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dil_size + 1, 2 * dil_size + 1),
		Point(dil_size, dil_size));

	Mat dil;
	dilate(img_temp, dil, element);

	// Search for the different polygons in picture
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dil, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Select the two biggest area
	int largestArea = 0, largestAreaIndex = 0, secondLargestArea = 0, secondLargestAreaIndex = 0;
	Rect boundRect1, boundRect2;
	for (int i = 0; i< contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largestArea) {
			Rect temp = boundingRect(contours[i]);
			if (temp.br().y - temp.tl().y >= TOLERANCE_VALUE) {
				if (hierarchy[i][3] == -1) {
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
		}
		else if (a > secondLargestArea) {
			Rect temp = boundingRect(contours[i]);
			if (temp.br().y - temp.tl().y >= TOLERANCE_VALUE) {
				if (hierarchy[i][3] == -1) {
					// a = secondLargest
					secondLargestArea = a;
					secondLargestAreaIndex = i;
					boundRect2 = boundingRect(contours[i]);
				}
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
		leftPicture = getPolygon(contours, leftAreaIndex, hierarchy, img_temp, thresholdValue);
		//imshow("left", leftPicture.image);
	}
	// Set right picture
	if (rightArea == 0) {
		rightPicture.image = NULL;
		rightPicture.insideContourNumber = 0;
	}
	else {
		rightPicture = getPolygon(contours, rightAreaIndex, hierarchy, img_temp, thresholdValue);
		//imshow("right", rightPicture.image);
	}
}

/*
* Function which return the polygon correspondant to the contour passed in parameters
* It also rotate polygon to have a straight number if bool is false
*/
Picture getPolygon(vector<vector<Point>> contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src, int thresholdValue) {

	// Set the area of the number (with a straight rectangle)
	vector<vector<Point>> contours_poly(contours.size());
	approxPolyDP(Mat(contours[areaIndex]), contours_poly[areaIndex], 3, true);
	Rect boundRect = boundingRect(Mat(contours_poly[areaIndex]));

	// Draw this area in a empty matrix (with picture source's size)
	Scalar polygonColor = Scalar(255, 255, 255); // White contour

	// Crop picture source picture
	cropPicture(img_src, boundRect);

	// Get straight polygon (a biggest picture to not lose parts of original picture)
	Mat straightPicture = getStraightPolygon(img_src, boundRect, contours[areaIndex], areaIndex);
	// Picture manipulation on new picture
	//cvtColor(straightPicture, straightPicture, CV_BGR2GRAY);
	//if (INVERSE_PICTURE)
	//	straightPicture = inverseColor(straightPicture);



	//blur(straightPicture, straightPicture, Size(3, 3));
	//threshold(straightPicture, straightPicture, thresholdValue, 255, THRESH_BINARY);
	// Threshold determined with otsu's method
	//threshold(straightPicture, straightPicture, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

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
			Rect temp = boundingRect(contours_straight[i]);
			if (temp.br().y - temp.tl().y >= TOLERANCE_INTERN_VALUE) {
				approxPolyDP(Mat(contours_straight[i]), contours_poly_straight[i], 3, true);
				drawContours(resPicture.image, contours_poly_straight, i, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());
				index++;
			}
		}
	}
	resPicture.insideContourNumber = index;

	// Crop picture
	cropPicture(resPicture.image, boundRectStraight);

	return resPicture;
}

/*
* Function to get rotate polygon to have the polygon straight
*/
Mat getStraightPolygon(Mat img_src, Rect pictureBoundRect, vector<Point> pictureContour, int areaIndex) {

	int marge = 30;
	RotatedRect rotRect = minAreaRect(pictureContour);
	Point2f centerOfPicture = Point2f(pictureBoundRect.width / 2.0F, pictureBoundRect.height / 2.0F);
	Mat rotMat = getRotationMatrix2D(centerOfPicture, angleConversion(rotRect.angle), 1);

	//Scalar backgroundColor = Scalar::all(255);
	//if (INVERSE_PICTURE)
		Scalar backgroundColor = Scalar::all(0);

	Mat img_res(img_src.rows + 2 * marge, img_src.cols + 2 * marge, img_src.type(), backgroundColor);
	img_src.copyTo(img_res(Rect(marge, marge, img_src.cols, img_src.rows)));

	warpAffine(img_res, img_res, rotMat, img_res.size(), INTER_CUBIC);

	//Crop again around number
	/*vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_res, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point>> contours_poly(contours.size());
	approxPolyDP(Mat(contours[areaIndex]), contours_poly[areaIndex], 3, true);
	Rect boundRect = boundingRect(Mat(contours_poly[areaIndex]));
	cropPicture(img_res, boundRect);*/
	return img_res;
}

/*
* Function to get external contour
*/
vector<Point> getLargestContour(Mat pictureToCompare) {
	int marge = 10;
	Scalar backgroundColor = Scalar::all(0);
	Mat img_res(pictureToCompare.rows + 2 * marge, pictureToCompare.cols + 2 * marge, pictureToCompare.type(), backgroundColor);
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
	vector<vector<Point> > contours_res(contours.size());
	approxPolyDP(Mat(contours[largestAreaIndex]), contours_res[largestAreaIndex], 3, true);
	return contours_res[largestAreaIndex];
	//return contours[largestAreaIndex];
}

/*
* Function to crop a picture
*/
void cropPicture(Mat &pictureToCrop, Rect newArea) {

	IplImage *frame = new IplImage(pictureToCrop);
	cvSetImageROI(frame, cvRect(newArea.tl().x, newArea.tl().y, newArea.br().x - newArea.tl().x, newArea.br().y - newArea.tl().y));
	pictureToCrop = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);
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

/*
* Resize points of polygon with new dimensions
*/
void resizePoints(vector<Point> &points, int oldWidth, int oldHeight, int newWidth, int newHeight) {

	for (int i = 0; i < points.size();++i) {
		points[i].x = points[i].x * newWidth / oldWidth;
		points[i].y = points[i].y * newHeight / oldHeight;
	}
}
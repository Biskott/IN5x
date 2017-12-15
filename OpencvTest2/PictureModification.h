#pragma once

#include "Include.h"
#include "Constants.h"
#include "Angle.h"

// Picture modification
void setDefaultSearchingArea(Mat &img_src);
void setSearchingArea(Mat &img_src, Point a, Point b);
void pictureToPolygons(Mat img_src, Picture &leftPicture, Picture &rightPicture, int thresholdValue);
Picture getPolygon(vector<vector<Point>> contours, int areaIndex, vector<Vec4i> hierarchy, Mat img_src, int thresholdValue);
Mat getStraightPolygon(Mat img_src, Rect pictureBoundRect, vector<Point> pictureContour);
vector<Point> getLargestContour(Mat pictureToCompare);
void cropPicture(Mat &pictureToCrop, Rect newArea);
Mat inverseColor(Mat picture);
void resizePoints(vector<Point> &points, int oldWidth, int oldHeight, int newWidth, int newHeight);
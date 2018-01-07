#pragma once

#include "Include.h"
#include "Constants.h"
#include "PictureModification.h"
#include "Array.h"

// Character recognition
int applyKNN(Mat pictureToCompare);
vector<Point2i> KNNRange(Mat pictureToCompare);
vector<int> getAuthorizedNumbers(int insideContourNumber);
Point2i averageNumberFound(vector<Point2i> knnRange, vector<int> authorizedNumber);

// Main function
int getNumberInPicture(Mat pictureToCompare, float &percentage);
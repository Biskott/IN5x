#pragma once

#include "Include.h"
#include "Constants.h"
#include "PictureModification.h"
#include "Array.h"
#include "Matrix.h"

// Character recognition
Point2i applyKNN(Mat pictureToCompare, vector<int>);
vector<int> getAuthorizedNumbers(int insideContourNumber);

// Main function
int getNumberInPicture(Mat pictureToCompare, float &percentage);
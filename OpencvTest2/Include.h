#pragma once

// ------------------- INCLUDE ----------------------

#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
#include <cstdio>
#include <cstdlib>
//#include <sys/timeb.h>
#include <vector>
#include <Windows.h>

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace cv;


// ------------------- STRUCTURES ----------------------

struct Picture {
	Mat image;
	int insideContourNumber;
};


// ------------------- GLOBAL VARIABLES ----------------------

// Local variable for training's matrix
extern Mat TrainingImagesAsFlattenedFloats;
extern Mat ClassificationInts;
extern vector<Point2d> perimeterDatas;
extern map<int, vector<int>> insideContourTable;
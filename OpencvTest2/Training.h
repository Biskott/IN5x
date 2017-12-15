#pragma once

#include "Include.h"
#include "Constants.h"
#include "PictureModification.h"

// Training
void training();
bool saveXmlContourNumbers(map<int, set<int>> insideContourTable);
bool saveXmlPerimeter(vector<Point2d> perimeterDatas);
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened);
void setInDataBase(int value, Picture numberPicture, vector<int> &classificationIntsTab, Mat &trainingImagesAsFlattened, vector<Point2d> &perimeterDatas, map<int, set<int>> &insideContoursTable);
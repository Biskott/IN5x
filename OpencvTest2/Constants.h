#pragma once

#include "Include.h"

// ------------------- CONSTANTS FOR THE PROGRAM ----------------------

// Constants for camera size
const int CAMERA_PICTURE_HEIGHT = 480;
const int CAMERA_PICTURE_WIDTH = 640;

// Constants to resize pictures
const int RESIZED_IMAGE_WIDTH = 40;
const int RESIZED_IMAGE_HEIGHT = 60;

// Constant threshold for picture manipulation
const int THRESHOLD_VALUE = 190;

// Constant to set if the program made a new training 
const bool ASK_FOR_TRAINING_DEFAULT = true;

// Constant path
const string XML_PATH = "xml";
const string TRAINING_PICTURES_PATH = "TrainingPictures2";

// Constant filename
const string XML_PERIMETER_FILE_NAME = "perimeters.xml";
const string XML_CONTOUR_TABLE_FILE_NAME = "insideContourTable.xml";
const string XML_KNN_IMAGES_FILE_NAME = "images.xml";
const string XML_KNN_CLASSIFICATION_FILE_NAME = "classification.xml";

// Constant to filter size of polygons' detected
const int TOLERANCE_VALUE = 60;
const int TOLERANCE_INTERN_VALUE = 30;

// Search area in pourcentage of picture's size
const int SEARCH_AREA_LEFT = 30;
const int SEARCH_AREA_RIGHT = 60;
const int SEARCH_AREA_TOP = 30;
const int SEARCH_AREA_BOTTOM = 65;

// K parameter for KNN range algorithm : nb of nearest matches in range
const int KNN_K_PARAMETER = 1;

// Constant max value for percentage correspondance in KNN
const float MAX_KNN_VALUE = 4.6818E+8;
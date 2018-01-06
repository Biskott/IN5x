#include "stdafx.h"
#include "Training.h"
#include "Directory.h"

void read_directory(const string& name)
{
	string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(LPCWSTR(pattern.c_str()), &data)) != INVALID_HANDLE_VALUE) {
		do {
			cout << data.cFileName << endl;
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

/*
* Training function for the KNN algorithm
*/
void training() {

	// Variable to stock training's datas 
	Mat trainingImagesAsFlattened;
	vector<int> classificationIntsTab;
	vector<Point2d> perimeterDatas;
	map<int, set<int>> insideContoursTable;

	// List all the training pictures in the directory
	//vector<string> trainingPicturesNames = getFile(TRAINING_PICTURES_PATH + "\\*.png");  //Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);
	vector<string> trainingPicturesNames = getFile2();

	// Parse all pictures
	for (string s : trainingPicturesNames) {

		// Detect the number thanks to the file name
		int index = s.find_first_of('_');
		string number = s.substr(0, index);
		cout << "Training number read : " << number << endl;

		// Apply polygon detection
		Picture leftNumber, rightNumber; /*leftNumber.insideContourNumber = -1; rightNumber.insideContourNumber = -1;*/
		pictureToPolygons(imread(TRAINING_PICTURES_PATH + "\\" + s), leftNumber, rightNumber, THRESHOLD_VALUE);

		// Set values in training data's variable
		if (number.length() == 1) {
			if (!rightNumber.image.empty())
				setInDataBase(number[0] - 48, rightNumber, classificationIntsTab, trainingImagesAsFlattened, perimeterDatas, insideContoursTable);
		}
		else if (number.length() == 2) {
			if (!leftNumber.image.empty())
				setInDataBase(number[0] - 48, leftNumber, classificationIntsTab, trainingImagesAsFlattened, perimeterDatas, insideContoursTable);

			if (!rightNumber.image.empty())
				setInDataBase(number[1] - 48, rightNumber, classificationIntsTab, trainingImagesAsFlattened, perimeterDatas, insideContoursTable);
		}
	}

	// Save the two training xml files
	if (saveXmlTraining(classificationIntsTab, trainingImagesAsFlattened))
		cout << "Training xml files for KNN saved succesfully." << endl;

	// Save the xml perimeters file
	if (saveXmlPerimeter(perimeterDatas) && USE_PERIMETER) {
		cout << "Training xml file for perimeter range saved succesfully" << endl;
	}
	// Save the xml inside contour numbers file
	if (saveXmlContourNumbers(insideContoursTable)) {
		cout << "Training xml file for contour's numbers saved succesfully" << endl;
	}
}

/*
* Function to save xml training files for inside contours' numbers
*/
bool saveXmlContourNumbers(map<int, set<int>> insideContourTable) {


	FileStorage fsContourNumberDatas(XML_PATH + "\\" + XML_CONTOUR_TABLE_FILE_NAME, FileStorage::WRITE);

	if (fsContourNumberDatas.isOpened()) {

		for (int i = 0; i < insideContourTable.size(); ++i) {
			stringstream data, tag;
			tag << "contour" << i;
			set<int>::const_iterator sit(insideContourTable[i].begin()), send(insideContourTable[i].end());
			for (; sit != send; ++sit) {
				data << *sit;
			}
			fsContourNumberDatas << tag.str() << data.str();
		}
		fsContourNumberDatas.release();
		return true;
	}
	else {
		return false;
	}
}

/*
* Function to save xml training files for perimeter range
*/
bool saveXmlPerimeter(vector<Point2d> perimeterDatas) {

	FileStorage fsPerimeterDatas(XML_PATH + "\\" + XML_PERIMETER_FILE_NAME, FileStorage::WRITE);

	if (fsPerimeterDatas.isOpened()) {
		for (int i = 0; i < perimeterDatas.size(); ++i) {
			stringstream data, tag;
			tag << "perimeter" << i;
			data << perimeterDatas[i].x << ":" << perimeterDatas[i].y;
			fsPerimeterDatas << tag.str() << data.str();
		}
		fsPerimeterDatas.release();
		return true;
	}
	else {
		return false;
	}
}

/*
* Function to save xml training files for KNN algorithm
*/
bool saveXmlTraining(vector<int> classificationIntsTab, Mat trainingImagesAsFlattened) {

	// Save xml number list
	FileStorage fsClassification(XML_PATH + "\\" + XML_KNN_CLASSIFICATION_FILE_NAME, FileStorage::WRITE);
	if (fsClassification.isOpened()) {
		Mat classificationInts = Mat(classificationIntsTab);
		fsClassification << "classifications" << classificationInts;
		fsClassification.release();
	}
	else {
		return false;
	}
	// Save xml matrix's number list
	FileStorage fsTrainingImages(XML_PATH + "\\" + XML_KNN_IMAGES_FILE_NAME, FileStorage::WRITE);
	if (fsTrainingImages.isOpened()) {
		fsTrainingImages << "images" << trainingImagesAsFlattened;
		fsTrainingImages.release();
	}
	else {
		return false;
	}
	return true;
}

/*
* Function to add training picture's data in the different matrix
*/
void setInDataBase(int value, Picture numberPicture, vector<int> &classificationIntsTab, Mat &trainingImagesAsFlattened, vector<Point2d> &perimeterDatas, map<int, set<int>> &insideContoursTable)
{
	// Number list in order for KNN
	classificationIntsTab.push_back(value);

	// Inside contour
	insideContoursTable[numberPicture.insideContourNumber].insert(value);

	// Resize matrix for perimeter and KNN calculation
	Mat imageResized, imageFloat;
	resize(numberPicture.image, imageResized, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

	// Perimeter
	if (USE_PERIMETER) {
		if (PERIMETER_RESIZE) {
			vector<Point> resizePerimeter = getLargestContour(numberPicture.image);
			resizePoints(resizePerimeter, numberPicture.image.rows, numberPicture.image.cols, RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT);
			perimeterDatas.push_back(Point2d(value, arcLength(resizePerimeter, true)));
		}
		else {
			perimeterDatas.push_back(Point2d(value, arcLength(getLargestContour(numberPicture.image), true)));
		}
	}

	// KNN
	imageResized.convertTo(imageFloat, CV_32FC1);
	Mat imageFlattenedFloat = imageFloat.reshape(1, 1);
	trainingImagesAsFlattened.push_back(imageFlattenedFloat);
}
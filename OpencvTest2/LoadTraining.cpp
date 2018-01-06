#include "stdafx.h"
#include "LoadTraining.h"

/*
* Function to load inside contour values from xml file
*/
int loadInsideContourValues() {

	// File which contain the classification of numbers
	FileStorage fsContourNumberDatas(XML_PATH + "\\" + XML_CONTOUR_TABLE_FILE_NAME, FileStorage::READ);

	if (!fsContourNumberDatas.isOpened()) {
		cout << "Error, unable to open training inside contour file..." << endl;
		return -1;
	}
	insideContourTable = map<int, vector<int>>();

	int i = 0;
	string value = "Init";
	stringstream tag;
	while (!value.empty())
	{
		tag << "contour" << i;
		fsContourNumberDatas[tag.str()] >> value;

		insideContourTable[i] = vector<int>();
		for (char number : value) {
			insideContourTable[i].push_back(number - 48);
		}

		tag = stringstream();
		++i;
	}
	fsContourNumberDatas.release();

	if (i == 1) {
		cout << "Error, no datas in inside contours file..." << endl;
		return -2;
	}
	return 0;
}

/*
* Function to load perimeter values from xml file
*/
int loadPerimeterValues() {

	// File which contain the classification of numbers
	FileStorage fsPerimeterDatas(XML_PATH + "\\" + XML_PERIMETER_FILE_NAME, FileStorage::READ);

	if (!fsPerimeterDatas.isOpened()) {
		cout << "Error, unable to open training perimeter file..." << endl;
		return -1;
	}
	perimeterDatas = vector<Point2d>();

	int i = 0;
	string value = "Init";
	stringstream tag;
	while (!value.empty())
	{
		tag << "perimeter" << i;
		fsPerimeterDatas[tag.str()] >> value;

		int Index = value.find(':');
		if (!value.empty())
			perimeterDatas.push_back(Point2d(stod(value.substr(0, Index)), stod(value.substr(Index + 1, value.size() - Index - 1))));

		tag = stringstream();
		++i;
	}
	fsPerimeterDatas.release();

	if (i == 1) {
		cout << "Error, no datas in perimeter file..." << endl;
		return -2;
	}
	return 0;
}

/*
* Function to init KNN algorithm (load xml training files)
*/
int initKNN() {

	// File which contain the classification of numbers
	FileStorage fsClassifications(XML_PATH + "\\" + XML_KNN_CLASSIFICATION_FILE_NAME, FileStorage::READ);

	if (!fsClassifications.isOpened()) {
		cout << "Error, unable to open training classifications file..." << endl;
		return -1;
	}
	fsClassifications["classifications"] >> ClassificationInts;
	fsClassifications.release();

	// File which contain the picture's matrix of numbers
	FileStorage fsTrainingImages(XML_PATH + "\\" + XML_KNN_IMAGES_FILE_NAME, FileStorage::READ);

	if (fsTrainingImages.isOpened() == false) {
		cout << "Error, unable to open training images file..." << endl;
		return -2;
	}
	fsTrainingImages["images"] >> TrainingImagesAsFlattenedFloats;
	fsTrainingImages.release();

	// Check if files are not empty
	bool fileEmpty = false;
	if (ClassificationInts.empty()) {
		cout << "Error, no datas in classification file" << endl;
		fileEmpty = true;
	}
	if (TrainingImagesAsFlattenedFloats.empty()) {
		cout << "Error, no datas in images file" << endl;
		fileEmpty = true;
	}
	if (fileEmpty)
		return -3;
	return 0;
}
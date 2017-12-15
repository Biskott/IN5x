// ------------------- INCLUDE ----------------------

#include "stdafx.h"
#include "Constants.h"
#include "Camera.h"
#include "Training.h"
#include "LoadTraining.h"
#include "CharacterRecognition.h"
#include "PictureModification.h"
#include "TimeCount.h" 
#include "Directory.h"

/*#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"*/
using namespace cv;

// ------------------- HEADER ----------------------

void CallBackFunc(int event, int x, int y, int flags, void *userdata);
Mat img;
Point p1, p2;
bool initPointSet;

vector<int> applyShapeRecognition(Mat pictureToCompare);
void echanger(Point2d tableau[], int a, int b);
void quickSort(Point2d tableau[], int debut, int fin);

// ------------------- FUNCTIONS ----------------------

int main(int argc, char *argv[])
{
	int startTime;
	bool askForTraining = ASK_FOR_TRAINING_DEFAULT;

	// Set if training is required
	if (argc > 1) {
		if ((string)argv[1] == "true")
			askForTraining = true;
	}

	// Launch the training function
	if (askForTraining) {
		startTime = getMilliCount();
		cout << "--- Training started ---" << endl;
		training();
		cout << "--- Training finished (" << getMilliSpan(startTime) << " ms) ---" << endl << endl;
	}

	// Loading KNN xml
	startTime = getMilliCount();
	if(initKNN()==0)
		cout << "KNN database's xml loaded (" << getMilliSpan(startTime) << " ms)" << endl;

	// Loading xml perimeter
	if (USE_PERIMETER) {
		startTime = getMilliCount();
		if (loadPerimeterValues() == 0)
			cout << "Perimeters' xml loaded (" << getMilliSpan(startTime) << " ms)" << endl;
	}

	// Loading xml inside contours
	startTime = getMilliCount();
	if(loadInsideContourValues()==0)
		cout << "Inside contours' xml loaded (" << getMilliSpan(startTime) << " ms)" << endl << endl;

	// Main loop
	cout << "--- Main loop has started ---" << endl;

	p1 = Point(0, 0);
	p2 = Point(CAMERA_PICTURE_WIDTH, CAMERA_PICTURE_HEIGHT);

	while (true) {
		cout << endl;
		int ACTUAL_CAMERA_ID = 1;
		Mat image;
		char key = 'q';
		while (key != ' ' && key != 27 && key !=8 && key != 'a' && key != 't' && key != 'y' && key != 's') {
			image = takePicture(ACTUAL_CAMERA_ID);
			if(p1.x==0 && p1.y==0 && p2.x==CAMERA_PICTURE_WIDTH && p2.y==CAMERA_PICTURE_HEIGHT)
				setDefaultSearchingArea(image);
			else
				setSearchingArea(image, p1, p2);
			imshow("Image prise", image);
			key = waitKey(10);
		}
		if (key == 27)
			break;
		if (key == 8) {
			destroyAllWindows();
			img = takePicture(ACTUAL_CAMERA_ID);
			namedWindow("Set area");
			setMouseCallback("Set area", CallBackFunc, NULL);
			imshow("Set area", img);
			initPointSet = false;
			while (waitKey(10) != ' ') {}
			destroyAllWindows();
		}
		if (key == ' ') {
			startTime = getMilliCount();
			int getNumberValue = getNumberInPicture(image);
			cout << "Number read with full function : " << getNumberValue << " (" << getMilliSpan(startTime) << " ms)" << endl;
		}
		if (key == 'a') {
			cout << "Number : ";
			string value;
			cin >> value;
			imwrite(TRAINING_PICTURES_PATH + "\\" + value + "_picture" + ".png", image);
		}
		if (key == 'y') {

			for (int i = 0; i < 5; ++i) {
				string name;
				stringstream ss;
				ss << i;
				imshow(ss.str(), takePicture(i));
			}
		}
		if (key == 's') {
			startTime = getMilliCount();
			vector<int> result = applyShapeRecognition(image);
			cout << "Value found with shape recognition : " << result[0] << " (" << getMilliSpan(startTime) << " ms)" << endl;
		}
	}
	

    return 0;
}

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {

	if (event == EVENT_LBUTTONDOWN) {
		p1 = Point(x, y);
		initPointSet = true;
	}
	if (event == EVENT_LBUTTONUP) {
		Mat new_img = img.clone();
		p2 = Point(x, y);
		rectangle(new_img, p1, p2, Scalar(0, 0, 255), 2, 8, 0);
		imshow("Set area", new_img);
		initPointSet = false;
	}
	if (event == EVENT_MOUSEMOVE && initPointSet) {
		Mat temp = img.clone();
		rectangle(temp, p1, Point(x, y), Scalar(0, 0, 255), 2, 8, 0);
		imshow("Set area", temp);
	}
}

vector<int> applyShapeRecognition(Mat pictureToCompare) {

	vector<int> ints;
	vector<vector<Point>> points;
	vector<Point2d> correspondance;
	
	vector<string> trainingPicturesNames = getFile(TRAINING_PICTURES_PATH + "\\*.png");

	for (string s : trainingPicturesNames) {

		int index = s.find_first_of('_');
		string number = s.substr(0, index);

		Mat img = imread(TRAINING_PICTURES_PATH + "\\" + s);
		Picture L, R;
		pictureToPolygons(img, L, R, THRESHOLD_VALUE);

		if (number.length() == 1) {
			if (!R.image.empty()) {
				ints.push_back(number[0]);
				points.push_back(getLargestContour(R.image));
			}
		}
		else if (number.length() == 2) {
			if (!L.image.empty())
			{
				ints.push_back(number[0]);
				points.push_back(getLargestContour(L.image));
			}
			if (!R.image.empty())
			{
				ints.push_back(number[1]);
				points.push_back(getLargestContour(R.image));
			}
		}
	}
	Picture pict, temp;
	pictureToPolygons(pictureToCompare, pict, temp, THRESHOLD_VALUE);
	pictureToCompare = temp.image;

	for (int i = 0; i < points.size(); i++) {
		double a = matchShapes(points[i], getLargestContour(pictureToCompare), CV_CONTOURS_MATCH_I3, 0);
		correspondance.push_back(Point2d(ints[i] - 48, a));
	}

	//destroyAllWindows();

	imshow("source", pictureToCompare);

	quickSort(correspondance.data(), 0, correspondance.size() - 1);
	vector<int> res = vector<int>();
	for (int i = 0; i < 10; ++i) {
		res.push_back(correspondance[i].x);
		cout << correspondance[i].x << " " << correspondance[i].y << endl;
	}
	return res;
}

void echanger(Point2d tableau[], int a, int b) {
	Point2d temp = tableau[a];
	tableau[a] = tableau[b];
	tableau[b] = temp;
}

void quickSort(Point2d tableau[], int debut, int fin) {
	int gauche = debut - 1;
	int droite = fin + 1;
	const double pivot = tableau[debut].y;

	if (debut >= fin)
		return;

	while (1)
	{
		do droite--; while (tableau[droite].y > pivot);
		do gauche++; while (tableau[gauche].y < pivot);

		if (gauche < droite)
			echanger(tableau, gauche, droite);
		else break;
	}
	quickSort(tableau, debut, droite);
	quickSort(tableau, droite + 1, fin);
}
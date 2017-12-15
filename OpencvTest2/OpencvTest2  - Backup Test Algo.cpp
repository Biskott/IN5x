// ------------------- INCLUDE ----------------------

#include "stdafx.h"
#include "Constants.h"
#include "Camera.h"
#include "Training.h"
#include "LoadTraining.h"
#include "CharacterRecognition.h"
#include "PictureModification.h"
#include "TimeCount.h"



# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv2/nonfree/features2d.hpp"
using namespace cv;


// ------------------- HEADER ----------------------

void CallBackFunc(int event, int x, int y, int flags, void *userdata);
Mat img;
Point p1, p2;
bool initPointSet;

int testFunction();

Mat img2; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;
void MatchingMethod(int, void*);
int testFunction2();

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
		if (key == 't') {
			Mat black = imread("test\\black.png");
			Mat white = imread("test\\white.png");

			Mat trainingMat;
			Mat imageResized, imageFloat;
			resize(black, imageResized, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
			imageResized.convertTo(imageFloat, CV_32FC1);
			Mat imageFlattenedFloat = imageFloat.reshape(1, 1);
			trainingMat.push_back(imageFlattenedFloat);

			CvMat training = trainingMat;
			CvMat classification = Mat(vector<int> {1});

			CvKNearest knn(&training, &classification, 0, false, 4);
			CvMat* nearests = cvCreateMat(1, 1, CV_32FC1);

			if (white.empty())
				cout << "error, empty picture" << endl;

			Mat pictureFloat;
			resize(white, white, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
			white.convertTo(pictureFloat, CV_32FC1);
			pictureFloat = pictureFloat.reshape(1, 1);

			Mat currentChar(0, 0, CV_32F);

			CvMat test = pictureFloat;
			CvMat *dist = cvCreateMat(1, 1, CV_32FC1);
			float fitCurrentChar = knn.find_nearest(&test, 1, 0, 0, nearests, dist);
			for (int i = 0; i < 1; ++i) {
				cout << nearests->data.fl[i] << " : " << dist->data.fl[i] << " , " << dist->data.db[i] << " , " << dist->data.i[i] << " , " << dist->data.s[i] << " , " << dist->data.ptr[i] << endl;
			}
			vector<int> range;
			for (int i = 0; i < 1; ++i) {
				range.push_back(nearests->data.fl[i]);
			}

			cout << "Value found : " << range[0] << endl;
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
			applyShapeRecognition(image);
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

int testFunction() {

	//Mat img_object = imread("pictureToSearch.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_object = imread("TrainingPictures_blackMovement\\2_picture.png", CV_LOAD_IMAGE_GRAYSCALE);
	resize(img_object, img_object, Size(img_object.cols*0.8, img_object.rows*0.8));
	Mat img_scene = imread("full2.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img_scene = imread("fullPicture.png", CV_LOAD_IMAGE_GRAYSCALE);

	/*Picture a, b;
	pictureToPolygons(imread("TrainingPictures_blackMovement\\15_picture.png"), a, b, THRESHOLD_VALUE);
	Mat img_object = b.image;
	pictureToPolygons(imread("TrainingPictures_blackMovement\\25_picture.png"), a, b, THRESHOLD_VALUE);
	Mat img_scene = b.image;*/

	if (!img_object.data || !img_scene.data)
	{
		printf(" --(!) Error reading images \n"); return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 300;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect(img_object, keypoints_object);
	detector.detect(img_scene, keypoints_scene);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute(img_object, keypoints_object, descriptors_object);
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	//-- Localize the object from img_1 in img_2
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(img_object.cols, 0);
	obj_corners[2] = Point(img_object.cols, img_object.rows); obj_corners[3] = Point(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);


	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	Point2f offset((float)img_object.cols, 0);
	line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);

	waitKey(0);

	return 0;
}

int testFunction2()
{
	/*Picture a, b;
	pictureToPolygons(imread("TrainingPictures_blackMovement\\15_picture.png"), a, b, THRESHOLD_VALUE);
	img2 = b.image;
	pictureToPolygons(imread("TrainingPictures_blackMovement\\25_picture.png"), a, b, THRESHOLD_VALUE);
	templ = b.image;*/

	/// Load image and template
	img2 = imread("full4.png", 1);
	templ = imread("TrainingPictures_blackMovement\\2_picture.png", 1);

	/// Create windows
	namedWindow(image_window, CV_WINDOW_AUTOSIZE);
	namedWindow(result_window, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar
	char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

	MatchingMethod(0, 0);

	waitKey(0);
	return 0;
}

void MatchingMethod(int, void*)
{
	/// Source image to display
	Mat img_display;
	img2.copyTo(img_display);

	/// Create the result matrix
	int result_cols = img2.cols - templ.cols + 1;
	int result_rows = img2.rows - templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(img2, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);

	return;
}


vector<int> applyShapeRecognition(Mat pictureToCompare) {

	vector<int> ints;
	vector<vector<Point>> points;
	vector<Point2d> correspondance;

	vector<string> trainingPicturesNames = Directory::GetListFiles(TRAINING_PICTURES_PATH, "*.png", false);

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
		double a = matchShapes(points[i], getLargestContour(pictureToCompare), CV_CONTOURS_MATCH_I2, 0);
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
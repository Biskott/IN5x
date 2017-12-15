
using namespace std;
using namespace cv;

// Header
int testTakePicture();
Mat sobelDerivative(Mat img_src);
Mat pictureTransformation(Mat picture);
Mat inverseColor(Mat picture);
Mat contourDetectionPolygonDrawing(Mat img_src);
Mat contourDetectionDrawing(Mat img_src);
bool pictureNumberSegmentation(Mat img_src, Mat &imgs_res);
bool pictureNumberSegmentationPolygon(Mat img_src, Mat &img_res);
void cutPicture(Mat img_src, Mat &leftPicture, Mat &rightPicture);

/*
* Function to test the capture of a picture
*/
int testTakePicture()
{
	VideoCapture capture(1);
	cout << "Started processing - Capturing image" << endl;

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(CV_CAP_PROP_GAIN, 0);

	if (!capture.isOpened()) {
		cout << "Failed to connect to the camera" << endl;
	}
	Mat frame, gray, edges, edges2;
	capture >> frame;
	if (frame.empty()) {
		cout << "Failed to capture an image" << endl;
		return -1;
	}
	cout << "Processing - Performing Image processing" << endl;
	cvtColor(frame, gray, CV_BGR2GRAY);
	blur(gray, edges, Size(3, 3));
	Canny(edges, edges2, 10, 30, 3);
	cout << "Finished processing - saving" << endl;

	// Saving
	imwrite("capture.png", frame);
	imwrite("gray.png", gray);
	imwrite("edges.png", edges);
	imwrite("edges2.png", edges2);
	return 0;
}

/*
* Function to make a sobel derivative
*/
Mat sobelDerivative(Mat img_src) {

	Mat img_src2, img_gray, grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	if (!img_src.data) {
		return img_src;
	}

	GaussianBlur(img_src, img_src2, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(img_src2, img_gray, CV_BGR2GRAY);

	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

	Sobel(img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;
}

/*
* Function which realize a picture transformation without polygon trace
*/
Mat pictureTransformation(Mat picture) {

	picture = sobelDerivative(picture);

	threshold(picture, picture, 127, 255, THRESH_BINARY);

	//GaussianBlur(picture, picture, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	morphologyEx(picture, picture, MORPH_OPEN, element);

	return picture;
}

/*
* Function to make a negative of a picture
*/
Mat inverseColor(Mat picture) {

	Mat new_img = Mat::zeros(picture.size(), picture.type());

	Mat sub_mat = Mat::ones(picture.size(), picture.type()) * 255;

	subtract(sub_mat, picture, new_img);

	return new_img;
}

/*
* Function to draw polygons (numbers) detected
*/
Mat contourDetectionPolygonDrawing(Mat img_src)
{
	Mat img_res;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// Resize picture
	img_src = ResizePicture(img_src);

	cvtColor(img_src, img_res, CV_BGR2GRAY);
	blur(img_res, img_res, Size(3, 3));

	// Seuillage
	threshold(img_res, img_res, 127, 255, THRESH_BINARY);

	/// Find contours
	findContours(img_res, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	img_res = Mat::zeros(img_res.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		if (boundRect[i].br().y - boundRect[i].tl().y > TOLERANCE_VALUE) {
			Scalar polygonColor = Scalar(255, 255, 255); // White contour
			Scalar rectColor = Scalar(0, 0, 255);		// Red rectangle
			drawContours(img_res, contours_poly, i, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(img_res, boundRect[i].tl(), boundRect[i].br(), rectColor, 2, 8, 0);
		}
	}
	return img_res;
}

/*
* Function to draw the contour of the number detected
*/
Mat contourDetectionDrawing(Mat img_src) {

	Mat img_res;

	// Resize picture
	img_src = ResizePicture(img_src);

	cvtColor(img_src, img_res, CV_BGR2GRAY);
	blur(img_res, img_res, Size(3, 3));
	threshold(img_res, img_res, 127, 255, THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_res, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	for (int i = 0; i< contours.size(); i++)
	{
		if (boundRect[i].br().y - boundRect[i].tl().y > TOLERANCE_VALUE) {
			Scalar rectColor = Scalar(0, 0, 255);		// Red rectangle
			rectangle(img_src, boundRect[i].tl(), boundRect[i].br(), rectColor, 2, 8, 0);
		}
	}
	return img_src;
}

/*
* Function to return resized pictures of the different numbers detected
*/
bool pictureNumberSegmentation(Mat img_src, Mat &imgs_res) {

	Mat img_temp;
	int pixelMarge = 10;

	// Picture manipulation
	cvtColor(img_src, img_temp, CV_BGR2GRAY);
	blur(img_temp, img_temp, Size(3, 3));
	threshold(img_temp, img_temp, 127, 255, THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	IplImage *frame = new IplImage(img_src);
	for (int i = 0; i< contours.size(); i++)
	{
		if (boundRect[i].br().y - boundRect[i].tl().y > TOLERANCE_VALUE) {
			cvSetImageROI(frame, cvRect(boundRect[i].tl().x - pixelMarge, boundRect[i].tl().y - pixelMarge,
				boundRect[i].br().x - boundRect[i].tl().x + pixelMarge * 2, boundRect[i].br().y - boundRect[i].tl().y + pixelMarge * 2));
			imgs_res = (cvarrToMat(frame).clone());
			cvResetImageROI(frame);
			return true;
		}
	}
	return false;
}

/*
* Function to return resized pictures of the different polygons which correspond to the numbers detected
*/
bool pictureNumberSegmentationPolygon(Mat img_src, Mat &img_res) {

	Mat img_temp;

	// Picture manipulation
	cvtColor(img_src, img_temp, CV_BGR2GRAY);
	blur(img_temp, img_temp, Size(3, 3));
	threshold(img_temp, img_temp, 127, 255, THRESH_BINARY);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int largestArea = 0, largestAreaIndex = 0;
	Rect boundRect;
	for (int i = 0; i< contours.size(); i++)
	{
		double a = contourArea(contours[i], false);
		if (a > largestArea) {
			largestArea = a;
			largestAreaIndex = i;
			boundRect = boundingRect(contours[i]);
		}
	}
	vector<vector<Point> > contours_poly(contours.size());
	approxPolyDP(Mat(contours[largestAreaIndex]), contours_poly[largestAreaIndex], 3, true);
	boundRect = boundingRect(Mat(contours_poly[largestAreaIndex]));

	if (boundRect.br().y - boundRect.tl().y < TOLERANCE_VALUE) {
		return false;
	}

	img_temp = Mat::zeros(img_temp.size(), CV_8UC3);

	Scalar polygonColor = Scalar(255, 255, 255); // White contour
	drawContours(img_temp, contours_poly, largestAreaIndex, polygonColor, 1, 8, vector<Vec4i>(), 0, Point());

	IplImage *frame = new IplImage(img_temp);

	cvSetImageROI(frame, cvRect(boundRect.tl().x, boundRect.tl().y, boundRect.br().x - boundRect.tl().x, boundRect.br().y - boundRect.tl().y));
	img_res = (cvarrToMat(frame).clone());
	cvResetImageROI(frame);

	return true;
}

/*
* Function to divide a picture in two pictures
*/
void cutPicture(Mat img_src, Mat &leftPicture, Mat &rightPicture) {

	IplImage *frame = new IplImage(img_src);
	cvSetImageROI(frame, cvRect(0, 0, img_src.size().width / 2, img_src.size().height));
	leftPicture = (cvarrToMat(frame)).clone();
	cvResetImageROI(frame);

	cvSetImageROI(frame, cvRect(img_src.size().width / 2, 0, img_src.size().width, img_src.size().height));
	rightPicture = (cvarrToMat(frame)).clone();
	cvResetImageROI(frame);
}
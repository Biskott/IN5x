#include "stdafx.h"
#include "Camera.h"
#include "Constants.h"

/*
* Function to take a picture
*/
Mat takePicture(int cameraId) {

	VideoCapture capture(cameraId);

	capture.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_PICTURE_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_PICTURE_HEIGHT);
	capture.set(CV_CAP_PROP_GAIN, 0);

	if (!capture.isOpened()) {
		exit(1);
	}
	Mat picture;
	capture >> picture;
	if (picture.empty()) {
		exit(2);
	}
	return picture;
}
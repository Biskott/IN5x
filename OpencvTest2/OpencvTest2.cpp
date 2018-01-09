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

// ------------------- MAIN ----------------------

int main(int argc, char *argv[])
{
	int startTime;

	// Launch the training function
	if (ASK_FOR_TRAINING_DEFAULT) {
		startTime = getMilliCount();
		cout << "--- Training started ---" << endl;
		training();
		cout << "--- Training finished (" << getMilliSpan(startTime) << " ms) ---" << endl << endl;
	}

	// Loading KNN xml
	startTime = getMilliCount();
	if (initKNN() == 0)
		cout << "KNN database's xml loaded (" << getMilliSpan(startTime) << " ms)" << endl;

	// Loading xml inside contours
	startTime = getMilliCount();
	if (loadInsideContourValues() == 0)
		cout << "Inside contours' xml loaded (" << getMilliSpan(startTime) << " ms)" << endl << endl;

	// Main loop
	cout << "--- Reconnaissance des chiffres ---" << endl;

	vector<string> filenames = getFileNb(18);
	float averagePercentage = -1;
	int nbErro = 0;

	for (string name : filenames) {
		float percentage;
		Mat pictureToFind = imread(("TrainingPictures3\\" + name));
		destroyAllWindows();
		imshow("Picture searched", pictureToFind);
		waitKey();
		startTime = getMilliCount();
		int getNumberValue = getNumberInPicture(pictureToFind, percentage);
		if (getNumberValue == -1) {
			cout << "Error in detection" << endl;
		}
		else {
			cout << "Number read with full function : " << getNumberValue << " (" << getMilliSpan(startTime) << " ms)" << endl << endl;
			if (percentage == -1) averagePercentage = percentage;
			else averagePercentage += percentage;
		} 
	}
	cout << "--- End of detection ---" << endl;
	averagePercentage /= filenames.size();
	cout << endl << "Average percentage detected : " << averagePercentage << "%" << endl;
	cout << "Number of error during detection : " << nbErro << endl;
	cout << endl << "Press any key to quit the program." << endl;
	waitKey();
	return 0;
}
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
		startTime = getMilliCount();
		int getNumberValue = getNumberInPicture(pictureToFind, percentage);
		if (getNumberValue == -1) {
			cout << "Error in detection" << endl;
		}
		else {
			cout << "Number read with full function : " << getNumberValue << " (" << getMilliSpan(startTime) << " ms)" << endl;
			if (percentage == -1) averagePercentage = percentage;
			else averagePercentage += percentage;
		} 
	}
	averagePercentage /= filenames.size();
	cout << "Average percentage detected : " << averagePercentage << endl;

	while(true){} // Temporaire : pour laisser afficher la console
	return 0;
}